# Containers for Orca 1m model

## Build containers

```sh
apptainer build predictor.sif predictor.def
apptainer build evaluator.sif evaluator.def
```

## Run containers

```sh
apptainer shell  -B predictor_scripts/:/scripts predictor.sif
python ./predictor_scripts/predictor_API_clean_apptainer.py your_ip_address 5000

# in a separate terminal session
apptainer shell  -B evaluator_data/:/evaluator_data -B predictions:/predictions -B evaluator_scripts/:/scripts  evaluator.sif
python ./evaluator_scripts/evaluator_API_clean_apptainer.py your_ip_address 5000 /predictions
```

-------

### Extra notes:

  - In `evaluator_data/evaluator_message_orca_2seqs.json` the sequence input `"seq1": ["chr9", 110400000]` has format `seq_id: [chromosome, start_coordinate]`, the evaluator retrieves the sequence from start coordinate to start coordinate + 1M, and check if the sequence length is 1M before sending to predictor.
  - Right now, to calculate prediction and target correlations, the predictor explicitly takes the predictions from the first prediction task `predictor_json['prediction_tasks'][0]['predictions'][key]`. As right now the sequence inputs are independent from prediction tasks, and multiple predictions can be achieved by specifying multiple sequence inputs.
  - The predictor is set to use h1esc_1m model and evaluator using target_h1esc_1m. Change them to hff_1m and target_hff_1m, respectively, to use the hff model.

### Evaluator without target dataset

Evaluator contains target values from h1esc and hff and dataset. 
It downloads a 34G tar file, which can take a long time.

For fast build without target function, comment out the following lines (downloading this file) in evaluator.def
```sh
# target files, around 34G
wget https://zenodo.org/record/6234936/files/resources_mcools.tar.gz 
tar -xf resources_mcools.tar.gz --no-same-owner
rm resources_mcools.tar.gz
```

In `evaluator_scripts/evaluator_API_clean_apptainer.py` comment out the lines between "calculate Pearson correlation between prediction and target", to avoid comparison with the target.
