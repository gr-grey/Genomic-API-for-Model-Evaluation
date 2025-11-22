'''RESTful ORCA Evaluator Utilizing Predictor REST API'''

import os
import sys
import json
from requests.exceptions import RequestException, HTTPError

import config
from data_loader import load_and_validate_data
from evaluator_content_handler import negotiate_formats, get_predictions, deserialize_response
import evaluator_metrics_calculator


def run_evaluator(predictor_ip, predictor_port, output_dir):
    """
    Preprocesses the data, sends request, receives response,
    saves the response, and triggers metric calculation.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' did not exist. Created it successfully!")

    # Load and validate evaluator input data
    data_dict = load_and_validate_data()

    # Number of sequences we expect predictions for
    total_sequences = len(data_dict["sequences"])

    predictor_url = f"http://{predictor_ip}:{predictor_port}"
    response_payload = None
    is_success_response = False
    resp_fmt = None  # negotiated response MIME type

    try:
        # ---- 1) Negotiate wire formats with predictor (/formats) ----
        req_fmt, resp_fmt = negotiate_formats(predictor_url)
        # req_fmt: what we will send (e.g. application/json or application/msgpack)
        # resp_fmt: what we expect back (e.g. application/msgpack or application/json)

        # ---- 2) Send prediction request (/predict) ----
        response = get_predictions(predictor_url, data_dict, req_fmt, resp_fmt)

        # If we got here, requests.raise_for_status() did NOT raise HTTPError
        is_success_response = True
        print("Predictor returned 200 OK.")

        # ---- 3) Decode response in negotiated format ----
        response_payload = deserialize_response(response, resp_fmt)

    except HTTPError as http_err:
        # HTTP 4xx/5xx: treat as error response from predictor
        print(f"Predictor returned HTTP {http_err.response.status_code}. Processing error payload...")
        is_success_response = False

        # If negotiation failed before setting resp_fmt, fall back to JSON for error body
        fmt_for_error = resp_fmt or "application/json"
        try:
            response_payload = deserialize_response(http_err.response, fmt_for_error)
        except ValueError as decode_err:
            # Couldn’t decode the error body at all
            print(f"Could not decode the error response body: {decode_err}", file=sys.stderr)
            response_payload = {
                "predictor_name": "UnknownPredictor_ErrorResponse",
                "error": [{
                    "server_error": (
                        f"Failed to decode error response (Status {http_err.response.status_code}). "
                        f"Body (truncated): {http_err.response.text[:500]}..."
                    )
                }]
            }

    # At this point, any *data* / decode problems from deserialize_response()
    # will have been raised as ValueError and caught by __main__, printing
    # "FATAL ERROR (Data): ...". That behaviour is preserved.

    if response_payload is None:
        print("FATAL: No response payload received or processed.", file=sys.stderr)
        # Fallback synthetic payload
        response_payload = {
            "predictor_name": "UnknownPredictor_NoResponse",
            "error": [{
                "evaluator_error": "No response payload could be processed after request."
            }]
        }

    # ---- 4) Save the raw predictions/error payload ----
    predictor_name = response_payload.get("predictor_name", "UnknownPredictor").replace(" ", "_")
    output_filename = f"{config.output_filename_base}_from_{predictor_name}.json"
    saved_predictions_path = os.path.join(output_dir, output_filename)

    # Check sequence counts in each prediction task (if any)
    for i, task in enumerate(response_payload.get("prediction_tasks", []), start=1):
        preds = task.get("predictions", {})
        # preds is typically a dict of seq_id -> array; you may want len(preds)
        if isinstance(preds, dict):
            n_preds = len(preds)
        else:
            n_preds = len(preds)
        if n_preds != total_sequences:
            print(
                f"Warning: Task {i} ('{task.get('name')}') has {n_preds} predictions, "
                f"but {total_sequences} sequences were sent to the Predictor."
            )

    try:
        with open(saved_predictions_path, 'w', encoding='utf-8') as f:
            json.dump(response_payload, f, ensure_ascii=False, indent=4)
        print(f"Raw predictions saved to {saved_predictions_path}")
    except IOError as e:
        print(f"FATAL: Could not save predictions to {saved_predictions_path}. {e}", file=sys.stderr)
        return

    # ---- 5) Compute metrics only if the predictor returned 200 OK ----
    if is_success_response:
        evaluator_metrics_calculator.calculate_and_save_metrics(response_payload, output_dir)
    else:
        print("Skipping metrics calculation because the Predictor did not return a 200 OK status.")


if __name__ == '__main__':
    # Check arguments
    if len(sys.argv) != 4:
        print(
            "Invalid arguments! Usage:\n"
            "  python evaluator_RestAPI.py <predictor_ip_address> <predictor_port> <mounted_output_directory>"
        )
        sys.exit(1)

    predictor_ip = sys.argv[1]
    predictor_port = int(sys.argv[2])
    output_dir_arg = sys.argv[3]

    try:
        run_evaluator(predictor_ip, predictor_port, output_dir_arg)
        print("Evaluation complete.")
        sys.exit(0)

    except (FileNotFoundError, ValueError) as e:
        # FileNotFoundError: bad input path
        # ValueError: invalid data or response (e.g. JSON/msgpack decode errors)
        print(f"FATAL ERROR (Data): {e}", file=sys.stderr)
        sys.exit(1)

    except RequestException as e:
        print(
            f"FATAL ERROR (Network): Could not connect to predictor at "
            f"http://{predictor_ip}:{predictor_port}. {e}",
            file=sys.stderr
        )
        sys.exit(1)

    except Exception as e:
        print(f"An unexpected fatal error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)