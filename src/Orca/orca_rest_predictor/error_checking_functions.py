'''Error Classes and Error Checking Functions for Predictor'''

import numpy as np

# ERROR CLASSES
# Error classes can be added here to easily keep track of error status codes

class APIError(Exception):
    """
    Base class for all custom API errors.
    This allows us to catch all our custom errors with a single handler.
    """
    def __init__(self, message, status_code, error_key):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_key = error_key
        
class BadRequestError(APIError):
    """
    For errors where the request is unacceptable (e.g. malformed JSON, missing keys).
    Corresponds to the 'bad_prediction_request' key.
    """
    def __init__(self, message="The request was unacceptable."):
        super().__init__(message, status_code=400, error_key='bad_prediction_request')

class PredictionFailedError(APIError):
    """
    For errors where the request was valid, but the model could not complete the prediction.
    Corresponds to the 'prediction_request_failed' key.
    """
    def __init__(self, message="The model prediction was incomplete."):
        super().__init__(message, status_code=422, error_key='prediction_request_failed')

class ServerError(APIError):
    """
    For backend issues (e.g. memory errors, unexpected crashes).
    Corresponds to the 'server_error' key.
    """
    def __init__(self, message="An unexpected issue occurred on the server."):
        super().__init__(message, status_code=500, error_key='server_error')

# ------------------------
# ERROR CHECKING FUNCTIONS
# ------------------------

# Model-specific check: "prediction_request_failed" error
def check_seqs_specifications(sequences, json_return_error_model):
    """
    Check that sequences conform to model specs.
    - Valid bases: "A", "T", "C", "G", "N"
    - No empty sequences
    """
    valid_bases = {"A", "T", "C", "G", "N"}
    for seq_id, seq in sequences.items():
        if not seq:
            json_return_error_model["prediction_request_failed"].append(
                f"sequence '{seq_id}' is empty"
            )
        
        invalid_chars = set(seq.upper()) - valid_bases
        if invalid_chars:
            json_return_error_model['prediction_request_failed'].append(
                f"sequence '{seq_id}' has invalid character(s): {invalid_chars}"
            )
    
    return json_return_error_model


def check_mandatory_keys(evaluator_keys, json_return_error):
    """
    Top-level mandatory keys in the request JSON.
    """
    mandatory_keys = ["readout", "prediction_tasks", "sequences"]  # NOTE: "request" removed
    evaluator_keys_set = set(evaluator_keys)
    missing = list(sorted(set(mandatory_keys) - evaluator_keys_set))
    if missing:
        json_return_error['bad_prediction_request'].append(
            f"The following mandatory top-level keys are missing from the JSON: {', '.join(missing)}"
        )
    return json_return_error


def check_key_values_readout(readout_value, json_return_error):
    readout_options = ["point", "track", "interaction_matrix"]

    if readout_value not in readout_options:
        json_return_error['bad_prediction_request'].append(
            "readout requested is not recognized. Please choose from "
            "['point', 'track', 'interaction_matrix']"
        )

    if not isinstance(readout_value, str):
        json_return_error['bad_prediction_request'].append("'readout' value should be a string")

    if isinstance(readout_value, list):
        json_return_error['bad_prediction_request'].append("'readout' should only have 1 value")

    return json_return_error


def check_prediction_task_mandatory_keys(prediction_tasks, json_return_error):
    for index, prediction_task in enumerate(prediction_tasks):
        mandatory_keys = ["name", "type", "cell_type", "species"]
        task_keys = set(prediction_task.keys())
        missing = list(sorted(set(mandatory_keys) - task_keys))
        
        if missing:
            task_identifier = prediction_task.get("name", f"at index {index}")
            error_msg = (
                f"Mandatory keys missing from prediction_task '{task_identifier}': "
                f"{', '.join(missing)}"
            )
            print(error_msg)
            json_return_error['bad_prediction_request'].append(error_msg)
            
    return json_return_error


def check_prediction_task_name(prediction_tasks, json_return_error):
    for prediction_task in prediction_tasks:
        name = prediction_task.get('name')
        if isinstance(name, list):
            json_return_error['bad_prediction_request'].append("'name' should only have 1 value")
        elif not isinstance(name, str):
            json_return_error['bad_prediction_request'].append("'name' value should be a string")

    return json_return_error


def check_prediction_task_type(prediction_tasks, json_return_error):
    """
    For ORCA, we support chromatin_conformation (plus any binding_* if ever used).
    """
    prediction_task_options = ["chromatin_conformation"]

    for prediction_task in prediction_tasks:
        print(prediction_task)
        t = prediction_task.get('type')

        if isinstance(t, list):
            json_return_error['bad_prediction_request'].append("'type' should only have 1 value")
            continue

        if not isinstance(t, str):
            json_return_error['bad_prediction_request'].append("'type' value should be a string")
            continue

        # Allow exact types and binding_* family if needed
        if t in prediction_task_options or t.startswith('binding_'):
            continue
        else:
            json_return_error['bad_prediction_request'].append(
                f"prediction type {t} is not recognized"
            )

    return json_return_error


def check_prediction_task_cell_type(prediction_tasks, json_return_error):
    for prediction_task in prediction_tasks:
        cell_type = prediction_task.get('cell_type')
        if isinstance(cell_type, list):
            json_return_error['bad_prediction_request'].append("'cell_type' should only have 1 value")
        elif not isinstance(cell_type, str):
            json_return_error['bad_prediction_request'].append("'cell_type' value should be a string")

    return json_return_error


def check_prediction_task_species(prediction_tasks, json_return_error):
    for prediction_task in prediction_tasks:
        species = prediction_task.get('species')
        if isinstance(species, list):
            json_return_error['bad_prediction_request'].append("'species' should only have 1 value")
        elif not isinstance(species, str):
            json_return_error['bad_prediction_request'].append("'species' value should be a string")

    return json_return_error


def check_prediction_task_scale(prediction_tasks, json_return_error):
    for prediction_task in prediction_tasks:
        if 'scale' in prediction_task:
            scale = prediction_task['scale']

            if isinstance(scale, list):
                json_return_error['bad_prediction_request'].append("'scale' should only have 1 value")
            else:
                prediction_scale_options = ["linear", "log"]

                if scale not in prediction_scale_options:
                    json_return_error['bad_prediction_request'].append(
                        "scale requested is not recognized. Please choose from ['log', 'linear']"
                    )

                if not isinstance(scale, str):
                    json_return_error['bad_prediction_request'].append("'scale' value should be a string")
        else:
            # No 'scale' key is allowed (matches template behavior)
            pass

    return json_return_error


def check_prediction_ranges(prediction_ranges, sequences, json_return_error):
    """
    Checks that prediction_ranges are formatted correctly.
    Now includes checks for positive integers and start <= end.
    """
    for key, value in prediction_ranges.items():
        
        if not isinstance(value, list):
            json_return_error['bad_prediction_request'].append(
                f"Values for '{key}' in 'prediction_ranges' must be in a list"
            )
            
        if not value:
            continue
        
        if len(value) != 2:
            json_return_error['bad_prediction_request'].append(
                f"Range array for '{key}' in 'prediction_ranges' must have 2 elements"
            )
        
        if not all(isinstance(num, int) for num in value):
            json_return_error['bad_prediction_request'].append(
                f"Values in '{key}' in 'prediction_ranges' must be integers"
            )
        
        start = value[0]
        end = value[1]
        
        if start < 0 or end < 0:
            json_return_error['bad_prediction_request'].append(
                f"Invalid range for '{key}' in 'prediction_ranges': indices must be positive. "
                f"Received [{start}, {end}]"
            )
            
        if start > end:
            json_return_error['bad_prediction_request'].append(
                f"Invalid range for '{key}' in 'prediction_ranges': start index ({start}) "
                f"cannot be greater than end index ({end}). Received [{start}, {end}]"
            )

        seq_len = len(sequences.get(key, ''))
        if start >= seq_len or end >= seq_len:
            if seq_len == 0:
                 err_msg = (
                     f"Invalid range for '{key}': cannot specify a range for a non-existent "
                     "or empty sequence."
                 )
            else:
                err_msg = (
                    f"Invalid range for '{key}': index is out of bounds. The maximum valid "
                    f"index for a sequence of length {seq_len} is {seq_len - 1}."
                )
            json_return_error['bad_prediction_request'].append(err_msg)
    
    return json_return_error


def check_seq_ids(prediction_ranges, sequences, json_return_error):
    """
    Check that keys in prediction_ranges match those in sequences.
    """
    if prediction_ranges.keys() != sequences.keys():
        json_return_error['bad_prediction_request'].append(
            "sequence ids in prediction_ranges do not match those in sequences"
        )
    return json_return_error


def check_key_values_upstream_flank(upstream_seq, json_return_error):
    if isinstance(upstream_seq, list):
        json_return_error['bad_prediction_request'].append("'upstream_seq' should only have 1 value")
    else:
        if not isinstance(upstream_seq, str):
            json_return_error['bad_prediction_request'].append("'upstream_seq' value should be a string")

    return json_return_error


def check_key_values_downstream_flank(downstream_seq, json_return_error):
    if isinstance(downstream_seq, list):
        json_return_error['bad_prediction_request'].append("'downstream_seq' should only have 1 value")
    else:
        if not isinstance(downstream_seq, str):
            json_return_error['bad_prediction_request'].append("'downstream_seq' value should be a string")

    return json_return_error