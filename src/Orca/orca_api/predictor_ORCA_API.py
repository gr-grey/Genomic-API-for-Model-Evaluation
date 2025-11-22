############# orca prediction function
import sys, torch, warnings
ORCA_PATH='/orca/'
USE_CUDA=torch.cuda.is_available()
sys.path.append(ORCA_PATH)

from selene_sdk.sequences import Genome
import orca_predict
orca_predict.load_resources(models=['1M'], use_cuda=USE_CUDA)
from orca_predict import h1esc_1m, hff_1m

############ orca prediction function
def orca_prediction(sequences, seq_ids):
    predictions = {}
    for seq, id in zip(sequences, seq_ids):
        sequence_encoded = Genome.sequence_to_encoding(seq)[None, :, :]

        model = h1esc_1m # change to hff_1m for hff model
        with warnings.catch_warnings(): # suppress warning
            warnings.simplefilter("ignore", category=UserWarning)
            pred = model(torch.FloatTensor(sequence_encoded).transpose(1, 2)) # pred shape [1, 1, 250, 250]
        #model return .np by default
        predictions[id] = pred[0][0].cpu().detach().numpy()
    return predictions


# predictor_API_clean_apptainer.py
import os
import sys
import json
import tqdm
import struct
import socket
import msgpack
#this patch allows the mspgpack-numpy
import msgpack_numpy as m
m.patch()
from error_message_functions_updated import *
#from api_preprocessing_utils import *

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Determine if running inside a container or not
if os.path.exists('/.singularity.d'):
    # Running inside the container
    HELP_FILE = "/predictor_container_apptainer/predictor_help_message.json"
else:
    # Running outside the container
    PREDICTOR_CONTAINER_DIR = os.path.dirname(SCRIPT_DIR)
    HELP_FILE = os.path.join(SCRIPT_DIR, 'predictor_help_message.json')


# Set buffer size for TCP
BUFFER_SIZE = 65536

# ------ ADDITION: Configuration for Wire-Format ------
SUPPORTED_REQUEST_FORMATS = [fmt.lower() for fmt in ["json", "msgpack"]] # Remove msgpack if not supported
SUPPORTED_RESPONSE_FORMATS = [fmt.lower() for fmt in ["msgpack", "msgpack_numpy"]] # JSON is always supported even when not mentioned

def send_payload(sock, payload_obj, wire_fmt):
    
    """
    Helper to pack and send JSON or MsgPack, prefix with 4-byte length, and send.
    
    Args:
        sock: Client socket
        payload_obj: Payload being sent to Evaluator
        wire_fmt: The format to send the payload_obj in
    
    Returns:
        None
    """
    
    try:
        if wire_fmt in ["msgpack", "msgpack_numpy"]:
            body = msgpack.packb(payload_obj, use_bin_type=True)
        else:
            body = json.dumps(payload_obj).encode("utf-8")
        # Length-prefix
        sock.sendall(struct.pack(">I", len(body)))
        sock.sendall(body)
    except socket.error as e:
        print(f"server_error: Error sending payload: {e}")
        sock.close()

def negotiate_format_with_evaluator(client_socket):
    
    """
    1. Send advert JSON with:
         - "predictor_supported_request_formats"    (what Predictor can RECEIVE)
         - "predictor_supported_response_formats" (what Predictor can SEND BACK)
    2. Read back Evaluator choice JSON with:
         - "request_format"    (what Evaluator will use to send)
         - "response_format" (what Evaluator expects back)
    3. Validate both against SUPPORTED_REQUEST_FORMATS 
       and SUPPORTED_RESPONSE_FORMATS, respectively
    
    Returns:
        Agreed (request_fmt, response_fmt) on success;
        (None, None) Send error JSON and close the 
        connection with Evaluator on failure.
    """
    
    # Advertise
    supported_fmts = {
        "predictor_supported_request_formats": SUPPORTED_REQUEST_FORMATS,
        "predictor_supported_response_formats": SUPPORTED_RESPONSE_FORMATS
        }
    supported_fmts_bytes = json.dumps(supported_fmts).encode('utf-8')
    client_socket.sendall(struct.pack(">I", len(supported_fmts_bytes)))
    client_socket.sendall(supported_fmts_bytes)
    print(f"Advertised formats: {supported_fmts}")
    
    # Evaluator decides what its request and response formats will be
    # based on what was advertised to it.
    # This time evaluator is reaching out to predictor to send its decision
    # on the negotiated formats so predictor can handle incoming and outgoing
    # payload accordingly. If the evaluator still somehow sent 
    # REQUEST and PREDICTION formats that Predictor does not support,
    # send error and close connection with that evaluator.
    
    # Receive choice length from Evaluator
    prefix = client_socket.recv(4)
    if not prefix:
        print("Evaluator disconnected before sending preferred format.")
        client_socket.close()
        return None, None
    choice_len = struct.unpack(">I", prefix)[0]
    
    choice_recv = b""
    while len(choice_recv) < choice_len:
        chunk = client_socket.recv(BUFFER_SIZE) # This can change to receive the exact data length
        if not chunk:
            print("Error: incomplete choice payload. Closing connection!")
            client_socket.close()
            return None, None
        choice_recv += chunk
    
    # Receive Evaluator choice and validate
    try:
        preferences = json.loads(choice_recv.decode("utf-8"))
        request_fmt = preferences["request_format"].lower()       # Evaluator -> Predictor
        response_fmt = preferences["response_format"].lower() # Predictor -> Evaluator
        print(f"Evaluator will send request(s) in: {request_fmt}")
        print(f"Evaluator excpects predictions in: {response_fmt}")
    except Exception as e:
        send_payload(client_socket,
                     {"error": "bad_payload -- cannot parse format choice"},
                     "json")
        print(f"Error parsing evaluator choice: {e}")
        client_socket.close()
        return None, None
    
    # If unsupported, send error back as JSON and close 
    # The client will close before reaching this but this 
    # is a server side-check, in case client doesn't.
    # Lastly, JSON is always accepted even in cases where
    # Predictor does not mention that in 
    # SUPPORTED_REQUEST_FORMATS and SUPPORTED_RESPONSE_FORMATS
    accept_request_format = (request_fmt == "json") or (request_fmt in SUPPORTED_REQUEST_FORMATS)
    accept_response_format = (response_fmt == "json") or (response_fmt in SUPPORTED_RESPONSE_FORMATS)
    
    if not accept_request_format or not accept_response_format:
        err = {
            "error": (
                f"Unsupported formats: request must be one of {SUPPORTED_REQUEST_FORMATS}, "
                f"prediction must be one of {SUPPORTED_RESPONSE_FORMATS}"
            )
        }
        send_payload(client_socket, err, "json")
        print(f"Unsupported choice (request={request_fmt}, prediction={response_fmt}); closing.")
        client_socket.close()
        return None, None
    
    return request_fmt, response_fmt

def recv_message_loop(client_socket):
    # --- Perform the one-time handshake ---
    request_fmt, response_fmt = negotiate_format_with_evaluator(client_socket)
    if request_fmt is None or response_fmt is None:
        print("Send/Receive wire-format negotiation failed.")
        print("Closing connection with this Evaluator!")
        return None
    
    # Step 1: Receive total bytes (length) of the Evaluator's request 
    # Step 2: Receive file from Evaluator
    
    # ---------------------- Receive Evaluator JSON ----------------------
    #This loop allows multiple JSON files to be sent over from one Evaluator

    while True:
        # Before receiving JSON from Evaluator
        # Receive length of the incoming JSON message (4-byte integer)
        # Can change to 8-byte integer by changing .recv(4) to .recv(8)
        # and replacing format string '>I' to '>Q'
        # Step 1
        try:
            # Step 1: Read length prefix
            msg_length = client_socket.recv(4)
            if not msg_length:
                print("No further message length received. Closing connection.")
                print("This message can also show up even if all of the requests were complete -- please confirm!")
                client_socket.close()
                break # Exit the loop if no message length is received

            # Unpack message length from 4 bytes
            msglen = struct.unpack('>I', msg_length)[0]
            print(f"Expecting {msglen} bytes of data from the Evaluator ({request_fmt}).")
            
            # Step 2: Now receive the actual payload in packets
            # Initialize data to store a new message on each iteration
            # Clear data_recv variable so multiple requests can be made
            data_recv = b'' # formerly, json_data_recv
            # Initialize the progress bar
            # Initialize the progress bar
            progress = tqdm.tqdm(range(msglen), unit="B", 
                                 desc="Receiving Evaluator Request(s)",
                                 unit_scale=True, unit_divisor=1024)
            try:
                while len(data_recv) < msglen:
                    packet = client_socket.recv(BUFFER_SIZE) # can change
                    if not packet:
                        print("Connection closed unexpectedly.")
                        break
                    data_recv += packet
                    progress.update(len(packet))
            finally:
                # Close the progress bar when done
                progress.close()

            
           # Verify if all of the data is received
            if len(data_recv) == msglen:
                print("Evaluator request received completely")
                pass
            else:
                print("Data received was incomplete or corrupted.")
                break
            
        except Exception as e:
            print(f"Error while receiving data: {e}")
            client_socket.close()
            break  # Break the loop on exception
        # ---------------------- Process Received File ----------------------
        # --- Decode incoming payload into dict ---
        # This is to standardize payload received in any wire_format
        # so it can go through error-checking
        try:
            if request_fmt == "msgpack":
                print(f"Unpacking {request_fmt} payload")
                evaluator_json = msgpack.unpackb(data_recv, raw=False)
            else:
                print(f"Unpacking {request_fmt} payload")
                evaluator_json = json.loads(data_recv.decode("utf-8"))
        except Exception as e:
            print(f"Error while decoding incoming payload: {e}")
            send_payload(client_socket, 
                         {"error": 
                             "bad_payload -- error while decoding incoming payload"},
                         "json")
            break
   
        print(evaluator_json['prediction_tasks'])
   # If only a "help" was requested return the predictor information file
        if evaluator_json['request'] == "help":
            # model builder should place help file in predictor folder
            print(f"Help requested! Sending {HELP_FILE}...")
            jsonResult_help = json.load(open(HELP_FILE))
            send_payload(client_socket, jsonResult_help, "json")
            client_socket.close()
            break
        
        # re-usable error checking functions
        # group these functions
        json_return_error = {'bad_prediction_request': []}
        json_return_error = check_mandatory_keys(evaluator_json.keys(), json_return_error)
        json_return_error = check_request(evaluator_json['request'], json_return_error)
        json_return_error = check_prediction_task_mandatory_keys(evaluator_json['prediction_tasks'], json_return_error)
        # if any of the mandatory keys are missing immediately return an error to the evaluator
        if any(json_return_error.values()) == True:
            print("Validation error; sending error JSON!")
            send_payload(client_socket, json_return_error, "json")
            client_socket.close()
            break
        else:
            json_return_error = check_key_values_readout(evaluator_json['readout'], json_return_error)
            json_return_error = check_prediction_task_name(evaluator_json['prediction_tasks'], json_return_error)
            json_return_error = check_prediction_task_type(evaluator_json['prediction_tasks'], json_return_error)
            json_return_error = check_prediction_task_cell_type(evaluator_json['prediction_tasks'], json_return_error)
            json_return_error = check_prediction_task_species(evaluator_json['prediction_tasks'], json_return_error)
            if 'prediction_ranges' in evaluator_json.keys():
                json_return_error = check_seq_ids(evaluator_json['prediction_ranges'], evaluator_json['sequences'], json_return_error)
                json_return_error = check_prediction_ranges(evaluator_json['prediction_ranges'], json_return_error)

            if 'upstream_seq' in evaluator_json.keys() or 'downstream_seq' in evaluator_json.keys():
                json_return_error = check_key_values_upstream_flank(evaluator_json['upstream_seq'], json_return_error)
            if 'downstream_seq' in evaluator_json.keys():
                json_return_error = check_key_values_downstream_flank(evaluator_json['downstream_seq'], json_return_error)

            # --- MODEL SPECIFIC: Ensure this Orca Predictor only supports homo_sapiens ---
                # Handle unsupported readouts
            readout_type = evaluator_json.get('readout')
            if readout_type in ["point", "track"]:
                print("Orca cannot handle 'point' or 'track' readout types. Exiting gracefully!")
                json_return_error = {'bad_prediction_request': 
                    ["Orca cannot process 'point' or 'track' readout types."]}
                send_payload(client_socket, json_return_error, "json")
                client_socket.close()
                print("Connection to client closed")
                break

            for task in evaluator_json['prediction_tasks']:
                if task.get('species', '').lower() != "homo_sapiens":
                    json_return_error['bad_prediction_request'].append(
                        f"This predictor only supports species: homo_sapiens. Received '{task.get('species')}' for task '{task.get('name')}'."
                    )
                    break

            for task in evaluator_json['prediction_tasks']:
                if task.get('type', '').lower() not in ["chromatin_conformation"]:
                    json_return_error['bad_prediction_request'].append(
                        f"This predictor only supports type: ['chromatin_conformation']. Received '{task.get('type')}' for task '{task.get('name')}'."
                    )
                    break

            for task in evaluator_json['prediction_tasks']:
                if task.get('scale', '').lower() not in ["log"]:
                    json_return_error['bad_prediction_request'].append(
                        f"This predictor only supports type: ['log']. Received '{task.get('scale')}' for task '{task.get('name')}'."
                    )
                    break
                        
            # if any errors were caught return them all to evaluator
            if any(json_return_error.values()) == True:
                print("Validation error; sending error JSON!")
                send_payload(client_socket, json_return_error, "json")
                client_socket.close()
                break
            
# ---------------------- %%%%%%%---------------
        # Extract sequences to predict
        # Check that the sequences meet model specifications
        # Otherwise do any other formatting required for the model

        # Create JSON to return
        json_return = {
            'request': evaluator_json['request'],
            'predictor_name': "ORCA_1M",
            'prediction_tasks':[]
        }

        # Loop through all the prediction tasks
        for prediction_task in evaluator_json['prediction_tasks']:
            
            current_prediction_task = {'name': prediction_task['name']}

            current_prediction_task['type_requested'] =  prediction_task['type']
            current_prediction_task ['type_actual']  = 'HI-C'

            current_prediction_task['cell_type_requested'] = prediction_task['cell_type']
            current_prediction_task['cell_type_actual'] =  'H1-ESC'

            current_prediction_task['scale_prediction_requested'] =  prediction_task['scale']
            current_prediction_task['scale_prediction_actual']  = 'log'

            current_prediction_task['species_requested']  = prediction_task['species']
            current_prediction_task['species_actual']  = 'homo_sapiens'

            # Add predictions dictionary to the JSON
            sequences = evaluator_json['sequences'].values()
            seq_ids = evaluator_json["sequences"].keys()
            model_predictions = orca_prediction(sequences, seq_ids)
            print(model_predictions)
            if response_fmt in ["json", "msgpack"]:
                #model_predictions = model_predictions.tolist()
                print("converting numpy output to lists for JSON parsing")
                model_predictions = {k: v.tolist() for k, v in model_predictions.items()}

            print("ORCA OUTPUT")
            print(len(model_predictions))

            first_key = next(iter(model_predictions))
            print("Type of first value:", type(model_predictions[first_key]))

            if isinstance(model_predictions[first_key], np.ndarray):
                print("Shape:", model_predictions[first_key].shape)
            else:
                print("First value is not a NumPy array.")

            current_prediction_task['predictions'] = model_predictions

            # Append results for current prediction task to the main JSON object
            json_return['prediction_tasks'].append(current_prediction_task)

        send_payload(client_socket, json_return, response_fmt)


def run_predictor():

    predictor_ip = sys.argv[1]
    predictor_port = int(sys.argv[2])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket to a specific address and port
    server.bind((predictor_ip, predictor_port))
    # listen for incoming connections
    server.listen(0)
    print(f"Listening on {predictor_ip}:{predictor_port}")

    # This loop allows the Predictor server to stay running so that different Evaluators can connect
    while True:
        try:
            print("Waiting for a Evaluator to connect")
            #.accept() blocks code here until an Evaluator connects
            client_socket, client_address = server.accept()
            print(f"Accepted connection from {client_address[0]}:{client_address[1]}")
            #once connected the JSON file receiving can begin
            recv_message_loop(client_socket)
        except Exception as e:
            print(f"Error accepting client: {e}")
            break
if __name__ == '__main__':
    run_predictor()
