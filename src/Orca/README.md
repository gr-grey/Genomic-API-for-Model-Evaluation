# Containers for Orca 1m model

## Build containers

```sh
apptainer build predictor.sif predictor.def
apptainer build evaluator.sif evaluator.def
```

## Run RestAPI containers

 1. copy folders `orca_rest_evaluator` and `orca_rest_predictor`
 
 2. predictor session

 ```
  apptainer shell -B orca_rest_predictor/:/orca_rest predictor.sif

  pip install flask msgpack msgpack_numpy requests

  python /orca_rest/predictor_RestAPI.py 0.0.0.0 5000
 ```

 3. evaluator session

 ```
  apptainer shell -B orca_rest_evaluator/evaluator_data/:/evaluator_data -B orca_rest_evaluator/predictions/:/predictions -B orca_rest_evaluator/:/evaluator_rest evaluator.sif
  pip install msgpack msgpack_numpy requests pandas
  python /evaluator_rest/evaluator_RestAPI.py 127.0.0.1 5000 /predictions
 ```