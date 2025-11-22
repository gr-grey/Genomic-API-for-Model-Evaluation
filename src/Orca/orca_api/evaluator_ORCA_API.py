# evaluator_API_clean_apptainer.py
import os
import sys
import json
import tqdm
import struct
import socket
#import msgpack
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from seqstr import seqstr
import msgpack
import msgpack_numpy as m
m.patch()
import pandas as pd
############# get target for orca
import torch, warnings
ORCA_PATH='/orca/'
sys.path.append(ORCA_PATH)
USE_CUDA=torch.cuda.is_available()

import orca_predict
orca_predict.load_resources(models=['1M'], use_cuda=USE_CUDA)

from orca_predict import h1esc_1m, hff_1m, target_h1esc_1m, target_hff_1m
from orca_predict import h1esc_1m, hff_1m
from scipy.stats import pearsonr
############# get target for orca
EVALUATOR_NAME = 'ORCA_TestChr9'
# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the input JSON file name
# input_json = "chr9_sequence_coordinates.json"
input_json = "evaluator_message_orca_2seqs.json"

# Determine if running inside a container or not
if os.path.exists("/.singularity.d"):
    # Running inside the container
    EVALUATOR_DATA_DIR = "/evaluator_data/"
    PREDICTIONS_DIR = "/predictions"
else:
    # Running outside the container
    EVALUATOR_CONTAINER_DIR = SCRIPT_DIR
    EVALUATOR_DATA_DIR = os.path.join(EVALUATOR_CONTAINER_DIR, "evaluator_data")
    PREDICTIONS_DIR = os.path.join(EVALUATOR_CONTAINER_DIR, "predictions")
    
EVALUATOR_INPUT_PATH = os.path.join(EVALUATOR_DATA_DIR, input_json)
#PREDICTIONS_DIR='/scratch/st-cdeboer-1/iluthra/game_apis/no_ctm_not_containerized/Orca_Ishika_edited/'
RETURN_FILE_PATH = os.path.join(PREDICTIONS_DIR, f"Orca_predictions_{input_json}")

# Validate input file path
if not os.path.exists(EVALUATOR_INPUT_PATH):
    print(f"Error: Input file '{EVALUATOR_INPUT_PATH}' does not exist.")
    sys.exit(1)

# Validate output directory
# output_dir = os.path.dirname(RETURN_FILE_PATH)
# if not os.path.exists(output_dir):
#     print(f"Error: Output directory '{output_dir}' does not exist.")
#     sys.exit(1)
    
# Set buffer size for TCP
BUFFER_SIZE = 65536

# Debug logs for validation
print(f"Using input JSON: {EVALUATOR_INPUT_PATH}")
print(f"Will save predictions to: {RETURN_FILE_PATH}")

# ------ ADDITION: Configuration for Wire-Format ------
REQUEST_FORMAT = "msgpack"
REQUEST_FORMAT = REQUEST_FORMAT.lower() # for case-insensitive matching

# Compute send format before connecting to Predictor
PREDICTION_FORMAT = "msgpack_numpy"
PREDICTION_FORMAT = PREDICTION_FORMAT.lower()

#function to check for duplicate keys in the JSON file
def check_duplicates(json_file_path):

    """
    Parses a JSON file to detect and report any duplicate keys at the same level in the same object.
    This function ensures that no keys are silently overwritten in dictionaries.

    The function uses a helper to track the number of times each key appears during parsing,
    leveraging the `object_pairs_hook` parameter of `json.load()` to intercept key-value pairs
    before they are processed into a dictionary. If duplicates are detected at any level, they
    are reported with their counts and paths. Keys reused in separate objects within arrays
    (e.g., lists) are not considered duplicates.

    Args:
        json_file_path (str): The path to the JSON file to parse and check for duplicates.

    Returns:
        None:
            - If no duplicates are found, returns None, prints "No duplicates found."
            - If duplicates are found, prints the duplicate keys and their counts and returns None.
    """

    # Initialize a dictionary to track duplicate keys and their counts
    duplicate_keys = {}

    # Helper function to detect duplicates during JSON parsing
    def detect_duplicates(pairs):

        """
        Detects duplicate keys during JSON parsing and counts occurrences of each key.

        This function intercepts the key-value pairs provided by `json.load` and ensures that
        duplicate keys are flagged. It constructs the dictionary normally but counts how often
        each key appears, recording any keys that occur more than once.

        Args:
            pairs (list of tuple): A list of key-value pairs at the current level of the JSON.

        Returns:
            dict: A dictionary created from the key-value pairs.
        """

        # Use a local Counter to count occurrences of keys at this level
        local_counts = Counter()
        result_dict = {}
        for key, value in pairs:
            # Increment the count for each key
            local_counts[key] += 1
            # If the key is a duplicate, record it in the duplicate_keys dictionary
            if local_counts[key] > 1:
                duplicate_keys[key] = local_counts[key]
            # Add the key-value pair to the resulting dictionary
            result_dict[key] = value
        return result_dict

    try:
        # Open and parse the JSON file, using the helper to track duplicates
        with open(json_file_path, 'r') as file:
            data = json.load(file, object_pairs_hook=detect_duplicates)

        # Report duplicates if any were found
        if duplicate_keys:
            print("Duplicate keys found:")
            for key, count in duplicate_keys.items():
                print(f"Key: {key}, Count: {count}")
            return None  # Indicate that the JSON contains duplicates
        else:
            print("No duplicates found.")
            return data

    except FileNotFoundError:
        # Handle the case where the file is not found
        print(f"File not found: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        # Handle invalid JSON format errors
        print(f"Invalid JSON in file '{json_file_path}': {e}")
        return None


def negotiate_format_with_predictor(connection):
    
    """
    1. Read the advertised formats from Predictor:
        - "predictor_request_formats"    (what Predictor can RECEIVE)
        - "predictor_prediction_formats" (what Predictor can SEND BACK)
    2. Choose send_format = REQUEST_FORMAT if in predictor_request_formats else "json"
    3. Choose recv_format = PREDICTION_FORMAT if in predictor_prediction_formats else 
    4. Send back {"request_format": send_format, "prediction_format": recv_format}
    
    Returns:
        Agreed (send_format, recv_format)
    """
    
    # Receive advert length from Predictor
    prefix = connection.recv(4)
    if not prefix:
        print("Failed to receive supported formats from Predictor.")
        sys.exit(1)    
    supported_fmt_len = struct.unpack(">I", prefix)[0]
    
    # Read the advert payload
    supported_fmt = b""
    while len(supported_fmt) < supported_fmt_len:
        chunk = connection.recv(BUFFER_SIZE)
        if not chunk:
            print("Could not receive Predictor's supported wire_format. Closing connection!")
            sys.exit(1)
        supported_fmt += chunk
    print(supported_fmt)
    # Parse JSON advert
    try:
        supported = json.loads(supported_fmt.decode("utf-8"))
        pred_request_fmts = [f.lower() for f in supported.get("predictor_supported_request_formats")]
        pred_prediction_fmts = [f.lower() for f in supported.get("predictor_supported_response_formats")]
    except Exception as e:
        print("Error: Could not parse Predictor's supported formats")
        sys.exit(1)
        
    # JSON should always be accepted
    if "json" not in pred_request_fmts:
        pred_request_fmts.append("json")
    if "json" not in pred_prediction_fmts:
        pred_prediction_fmts.append("json")
    print(f"Predictor can receive: {pred_request_fmts}")
    print(f"Predictor can send back: {pred_prediction_fmts}")
    
    # Decide request format having parsed what Predictor can support
    if REQUEST_FORMAT in pred_request_fmts:
        send_format = REQUEST_FORMAT
    else:
        send_format = "json"
        if REQUEST_FORMAT != "json":
            print(f"WARNING: REQUEST_FORMAT='{REQUEST_FORMAT}' not supported by Predictor; Using JSON")
    
    # Decide prediction format
    if PREDICTION_FORMAT in pred_prediction_fmts:
        recv_format = PREDICTION_FORMAT
    else: 
        recv_format = "json"
        if PREDICTION_FORMAT != "json":
            print(f"WARNING: PREDICTION_FORMAT='{PREDICTION_FORMAT}' not supported by Predictor; Using JSON")
    
    # Send Evaluator decision back
    choice = json.dumps({
        "request_format": send_format,
        "response_format": recv_format
        }).encode('utf-8')
    connection.sendall(struct.pack(">I", len(choice)))
    connection.sendall(choice)
    print(f"Negotiated send format: {send_format}")
    print(f"Negotiated receive format: {recv_format}")
    return send_format, recv_format


def run_evaluator():
    host = sys.argv[1]
    port = int(sys.argv[2])
    output_dir = sys.argv[3]
    
    # Validate input JSON file
    if not os.path.exists(EVALUATOR_INPUT_PATH):
        print(f"Error: Evaluator input file '{EVALUATOR_INPUT_PATH}' does not exist.")
        sys.exit(1)

    # Validate output directory
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)
        
    # Try creating a socket
    try:
        # create a socket object
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error as e:
        print ("server_error: Error creating socket: %s" % e)
        sys.exit(1)
    # Re-try Parameters
    RETRY_INTERVAL = 300 # 300 seconds (5 mins)
    MAX_RETRIES = 5
    attempt = 0
    connected = False
    while attempt < MAX_RETRIES and not connected:
        try:
            # establish connection with predictor server
            connection.connect((host, port))
            print(f"Connected to Predictor on {host}:{port}")
            connected = True
        except socket.gaierror as e:
            print ("Address-related error connecting to server: %s" % e)
            sys.exit(1)
        except socket.error as e:
            attempt += 1
            print ("server_error: Connection error: %s" % e)
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_INTERVAL/60:.0f} minutes... (Attempt {attempt} of {MAX_RETRIES})")
                for _ in tqdm(range(RETRY_INTERVAL), desc="Waiting to retry connection", unit="s"):
                    time.sleep(1)
            else:
                print(f"Tried connecting {attempt} times. Exceeded maximum number of retries. Exiting...")
                sys.exit(1)
    # Negotiate wire format
    send_fmt, recv_fmt = negotiate_format_with_predictor(connection)

    try:
        # load in JSON file from evalutor_data if Predictor container was successful
        jsonResult = check_duplicates(EVALUATOR_INPUT_PATH)
        if jsonResult is None:
            sys.exit(1)
        # get sequence from seqstr
        seq_dict = jsonResult['sequence_coordinates']
        retrieved_seqs = {}
        seq_len = 1000000
        for key, val in seq_dict.items():
            chr, coord = val
            seqstr_input = f"[hg38]{chr}:{coord}-{coord+seq_len} +"
            print(f"fetching sequence: {seqstr_input}")
            seqstrout = seqstr(seqstr_input)
            seq = seqstrout[0].Seq
            if len(seq) == seq_len: # 1M model
                retrieved_seqs[key] = seq
            else:
                print(f"Sequence length does not match {seq_len}!")
        print(retrieved_seqs.keys())
        jsonResult['sequences'] = retrieved_seqs

        # Prepare payload -- Serialize
        print(f"Serializing request to Predictor as '{send_fmt}'")
        if send_fmt == "msgpack":
            try:
                payload_bytes = msgpack.packb(jsonResult, use_bin_type=True)
                print(f"Sending payload serialized as MsgPack")
            except Exception as e:
                print(f"Error packing MsgPack: {e}")
                sys.exit(1)
        else:
            try:
                payload_bytes = json.dumps(jsonResult).encode("utf-8")
                print(f"Sending payload serialized as JSON")
            except Exception as e:
                print(f"Error packing JSON: {e}")
                sys.exit(1)

        #jsonResult = json.dumps(jsonResult)
    except json.JSONDecodeError as e:
        print("Invalid JSON syntax:", e)

    # first send the total bytes we are transmitting to the Predictor
    # This is used to stop the recv() process
    # send the evaluator json to the predictor server
    try:
        payload_bytes_len = len(payload_bytes)
        connection.sendall(struct.pack(">I", payload_bytes_len))
        print(f"Sent evaluator request length {payload_bytes_len} bytes!")
        
        # Now send the actual payload
        connection.sendall(payload_bytes)
    except socket.error as e:
        print (f"server_error: Error sending payload: {e}")
        sys.exit(1)

# ---------------------- %%%%%%%---------------
    # receive message from the server
    data_recv = b''
    while True:
        # Before receiving JSON from Predictor
        # Receive length of the incoming JSON message (4-byte integer)
        # Can change to 8-byte integer by changing .recv(4) to .recv(8)
        # and replacing format string '>I' to '>Q'
        # Step 1
        try:
            msg_length = connection.recv(4)
            if not msg_length:
                print("Failed to receive message length. Closing connection.")
                connection.close()
                break # Exit the loop if no message length is received

            # Unpack message length from 4 bytes
            msglen = struct.unpack('>I', msg_length)[0]
            print(f"Expecting {msglen} bytes of data from the Predictor.")
            # Can comment out print commands other than for errors

            # Initialize the progress bar
            progress = tqdm.tqdm(range(msglen), unit="B",
                                desc="Receiving Predictor Response",
                                unit_scale=True)


            # Step 2: the payload
            # Now we want to receive the actual JSON in packets
            while len(data_recv) < msglen:
                packet = connection.recv(BUFFER_SIZE)
                if not packet:
                    print("Connection closed unexpectedly.")
                    break
                data_recv += packet
                progress.update(len(packet))
           
            # Close the progress bar when done
            progress.close()
            
            # Decode data if all of it is received
            if len(data_recv) == msglen:
                print("Predictor response received completely!")
                break
            else:
                print("Data received was incomplete or corrupted.")
                break


        except socket.error as e:
            print ("server_error: Error receiving predictions: %s" % e)
            sys.exit(1)

    # Parse and save Predictor response
    try:
        if recv_fmt == "msgpack":
            try:
                print("De-serializing Predictor response as MsgPack")
                predictor_data = msgpack.unpackb(data_recv, raw=False)
            # But in case of an error/help, Predictor will return JSON
            # Even if the agreed wire_fmt was not JSON
            except Exception:
                print("Error/ Help was received!")
                print("De-serializing Predictor response as JSON")
                predictor_data = json.loads(data_recv.decode("utf-8"))
        
        elif recv_fmt == "msgpack_numpy":
            try:
                print("Predictor response is in MsgPack format with numpy arrays")
                predictor_data = msgpack.unpackb(data_recv, raw=False)
            except Exception:
                print("Error/ Help was received!")
                print("De-serializing Predictor response as JSON")
                predictor_data = json.loads(data_recv.decode("utf-8"))
        else:
            try:
                print("De-serializing Predictor response as JSON")
                predictor_data = json.loads(data_recv.decode("utf-8"))
                #print(predictor_data)
            

            except (json.JSONDecodeError, IOError) as e:
                print(f"Error saving predictions: {e}")
                sys.exit(1)
        
        output_file = RETURN_FILE_PATH
        if recv_fmt == "msgpack_numpy":
            #this is the only way these files can be saved
            print("Saving Predictor response as .msgpack")
            with open(output_file + ".msgpack", "wb") as f:
                # Serialize the data and write it to the file
                msgpack.dump(predictor_data, f, use_bin_type=True)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictor_data, f,
                        ensure_ascii=False, indent=4, 
                        separators=(",", ": "))
            print(f"Predictions saved to {output_file}")

    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)

    try:
        if recv_fmt in ["msgpack", "json"]:
            print("----- Starting Evaluation Calculation and Saving as CSV -----")
            ############ calculate Pearson correlation between prediction and target
            correlations = {}
            model = h1esc_1m
            for key, val in seq_dict.items():
                chr, coord = val

                with warnings.catch_warnings(): # suppress runtime warning from printing to terminal
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    target = target_h1esc_1m.get_feature_data(chr, coord, coord + seq_len)[None, :, :]
                    # to bin across 4 to calculate the correlation
                    level = 4 # 1m target resolution is 1k, bin and average them every 4k
                    start = 0
                    target_r = np.nanmean(np.nanmean(np.reshape(target[:,start:start+250*level,start:start+250*level],(target.shape[0],250,level,250,level)),axis=4),axis=2)
                    level = 1 # 1M model only has level 1 normmats
                    #normalization for model specific scaling - otherwise can't compare - returned from trained model
                    target_np = np.log((target_r+model.epss[level])/(model.normmats[level]+model.epss[level]))[0, :, :]

                valid = np.isfinite(target_np)
                pred_arr = np.array(predictor_data['prediction_tasks'][0]['predictions'][key])

                corr = pearsonr(pred_arr[valid], target_np[valid])[0]
                correlations[key] = corr
                print(f"{key} correlation: {corr}")

            #predictor_data['correlations'] = correlations
            print(correlations)
            mean_correlation = sum(correlations.values()) / len(correlations)
            print(mean_correlation) 
            # ADDITION: Construct file name after receiving predictor_name
            predictor_name_received = predictor_data.get("predictor_name", None)
            predictor_name = predictor_name_received.replace(" ", "_").replace("/", "_")
            output_filename = f"ORCA_test9_from_{predictor_name}.csv"
    
            # Get UTC timestamp for predictor_nam
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S.%f")
            # Compute the full RETURN_FILE_PATH using the provided output directory

            print(f"Will save predictions to: {PREDICTIONS_DIR}")
            #print(predictor_data)
            prediction_task_data_onlyinfo = [{k: v for k, v in predictor_data["prediction_tasks"][0].items() if k != "predictions"}]

            pearson_r = mean_correlation
            #add code to create the output file
            evaluation_output = pd.DataFrame([{'Evaluator_Name': EVALUATOR_NAME, 'Description': "ORCA (Test Set - Chr9)", 'Predictor_Name': predictor_name,  'Time_Stamp': timestamp, 'Metric': 'pearson_r', 'Value': str(pearson_r), 'Prediction_task(s)_data': prediction_task_data_onlyinfo}])
            evaluation_output.to_csv(PREDICTIONS_DIR + '/' + output_filename , sep = "\t")

        ############ calculate Pearson correlation between prediction and target
        else:
            print("----- Starting Evaluation Calculation and Saving as CSV for MSGPACK-numpy return -----")
            ############ calculate Pearson correlation between prediction and target
            correlations = {}
            model = h1esc_1m
            for key, val in seq_dict.items():
                chr, coord = val

                with warnings.catch_warnings(): # suppress runtime warning from printing to terminal
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    target = target_h1esc_1m.get_feature_data(chr, coord, coord + seq_len)[None, :, :]
                    # to bin across 4 to calculate the correlation
                    level = 4 # 1m target resolution is 1k, bin and average them every 4k
                    start = 0
                    target_r = np.nanmean(np.nanmean(np.reshape(target[:,start:start+250*level,start:start+250*level],(target.shape[0],250,level,250,level)),axis=4),axis=2)
                    level = 1 # 1M model only has level 1 normmats
                    #normalization for model specific scaling - otherwise can't compare - returned from trained model
                    target_np = np.log((target_r+model.epss[level])/(model.normmats[level]+model.epss[level]))[0, :, :]

                valid = np.isfinite(target_np)
                pred_arr = np.array(predictor_data['prediction_tasks'][0]['predictions'][key])
                #print(pred_arr)
                #print(target_np)
                if np.all(np.isnan(target_np)):
                    print(f"Skipping {key}: target is all NaNs")
                    #correlations[key] = np.nan  # or continue if you want to skip storing it
                    continue
                else:
                    corr = pearsonr(pred_arr[valid], target_np[valid])[0]
                    correlations[key] = corr
                    print(f"{key} correlation: {corr}")

                
            #predictor_data['correlations'] = correlations
            print(correlations)
            mean_correlation = sum(correlations.values()) / len(correlations)
            print(mean_correlation) 
            # ADDITION: Construct file name after receiving predictor_name
            predictor_name_received = predictor_data.get("predictor_name", None)
            predictor_name = predictor_name_received.replace(" ", "_").replace("/", "_")
            output_filename = f"ORCA_testchr9_from_{predictor_name}.csv"
    
    
            # Get UTC timestamp for predictor_nam
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S.%f")
            # Compute the full RETURN_FILE_PATH using the provided output directory

            print(f"Will save predictions to: {PREDICTIONS_DIR}")

            prediction_task_data_onlyinfo = [{k: v for k, v in predictor_data["prediction_tasks"][0].items() if k != "predictions"}]
            pearson_r = mean_correlation
            #add code to create the output file
            evaluation_output = pd.DataFrame([{'Evaluator_name': EVALUATOR_NAME, 'Description': "ORCA (Test Set - Chr9)", 'Predictor_name': predictor_name,  'Time_stamp': timestamp, 'Metric': 'pearson_r', 'Value': str(pearson_r), 'Prediction_task(s)_data': prediction_task_data_onlyinfo}])
            evaluation_output.to_csv(PREDICTIONS_DIR + '/' + output_filename , sep = "\t")

        ############ calculate Pearson correlation between prediction and target

    except Exception as e:
        print(f"Error saving correlations: {e}")
        sys.exit(1)

    # ---------------------- %%%%%%%---------------
    connection.close()
    print("Connection to server closed")
    #return output_file



run_evaluator()

