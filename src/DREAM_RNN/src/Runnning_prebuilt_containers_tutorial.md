# Running the DREAM-RNN container with a sample dataset

To run a test prediction using the DREAM-RNN container and sample Evaluator container:

1. Download the containers from Zenodo: https://zenodo.org/records/14861069

```bash
mkdir DREAMRNN
mkdir test_evaluator
```

```bash
cd DREAMRNN
wget https://zenodo.org/records/14861069/files/predictor.sif?download=1
mv predictor.sif?download=1 predictor.sif
```

``` bash
cd test_evaluator
wget https://zenodo.org/records/14861069/files/evaluator.sif?download=1
mv evaluator.sif?download=1 evaluator.sif
wget https://zenodo.org/records/14861069/files/evaluator_data.zip?download=1
mv evaluator_data.zip?include_deleted=0 evaluator_data
mkdir predictions
```

Note: if you run into issues downloading the `evaluator_data` folder you may need to manually download it off Zenodo. 

2. Get the IP Address of where the Predictor is running

Note: PORTs above 5000 are usually free to use

`hostname -I`

3. Start the DREAMRNN Predictor with the IP address and PORT arguments

`apptainer run predictor.sif HOST PORT`

Example:
`apptainer run predictor.sif 172.16.47.243 5000`

4. Start the test Evaluator 

```bash 
apptainer run \ 
    -B /path/to/evaluator_data:/evaluator_data  \
    -B /path/to/predictions:/predictions  \
    evaluator.sif HOST PORT /predictions
```
Example:
```bash 
apptainer run \ 
    -B /path/to/evaluator_data:/evaluator_data  \
    -B /path/to/predictions:/predictions  \
    evaluator.sif 172.16.47.243 5000 /predictions
```
apptainer run -B //arc/project/st-cdeboer-1/iluthra/API_genomic_model_evaluation/Running_prebuilt_containers/test_evaluator/evaluator_data:/evaluator_data -B //arc/project/st-cdeboer-1/iluthra/API_genomic_model_evaluation/Running_prebuilt_containers/test_evaluator/predictions:/predictions evaluator.sif 172.16.47.243 5000 /predictions

The `-B` mounts local directories so that the Evaluator container can read in the JSON file from a local folder and write the prediction to the locally created `/predictions` folder.

5. If the Evaluator-Prediction communication was successful a JSON file will be found in the `predictions/` folder.

Yay! You just completed a successful communication between the DREAMRNN model and a test sequence set with GAME :)
    