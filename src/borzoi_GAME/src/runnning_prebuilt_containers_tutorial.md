# Running the Borzoi container with a sample dataset

To run a test prediction using the Borzoi container and sample Evaluator container:

1. Download the containers from Zenodo: <https://zenodo.org/records/14861069>

```bash
mkdir BORZOI
mkdir test_evaluator
```

```bash
cd BORZOI
wget [LINK]
mv borzoi_human_predictor.sif?download=1 borzoi_human_predictor.sif
```

``` bash
cd test_evaluator
wget [LINK]
mv evaluator.sif?download=1 evaluator.sif
wget [LINK]
mv evaluator_data.zip?include_deleted=0 evaluator_data
mkdir predictions
```

**Note:** if you run into issues downloading the `evaluator_data` folder you may need to manually download it off Zenodo.

2. Get the IP Address of where the Predictor is running

Note: PORTs above 5000 are usually free to use: `hostname -I`

3. Start the BORZOI Predictor with the IP address and PORT arguments

`apptainer run --nv borzoi_human_predictor.sif HOST PORT`

Example:
`apptainer run --nv borzoi_human_predictor.sif 172.16.47.243 5000`

4. Start the test Evaluator

```bash
apptainer run \
    -B absolute/path/to/evaluator_data:/evaluator_data \
    -B absolute/path/to/predictions:/predictions \
    borzoi_evaluator.sif PREDICTOR_HOST PREDICTOR_PORT /predictions
```

Example:

```bash
apptainer run \
    -B absolute/path/to/evaluator_data:/evaluator_data \
    -B absolute/path/to/predictions:/predictions \
    borzoi_evaluator.sif 172.16.47.244 5000 /predictions
```

The `-B` mounts local directories so that the Evaluator container can read in the JSON file from a local folder and write the prediction to the locally created `/predictions` folder.

5. If the Evaluator-Prediction communication was successful a JSON file will be found in the `predictions/` folder.

Yay! You just completed a successful communication between the BORZOI model and a test sequence set with GAME.
