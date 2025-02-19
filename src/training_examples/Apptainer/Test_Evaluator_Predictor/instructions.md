### Basic instructions to create a test Evaluator and Predictor

### Creating an Evaluator that reads in some sample JSON files that include DNA sequences

Make sure that you have Apptainer installed. Additional details and installation instructions can be found here: https://apptainer.org/docs/user/main/quick_start.html

The evaluator container in this example will require 3 arguments in this order: HOST, PORT, OUTPUT_DIR.

To create a sample Evaluator using the scripts and data we provide for this example follow the instructions below:

1. Download the `evaluator_container_sample` folder

2. The `evaluator_python_only.def` is a definition file and will be used to create the Apptainer container. In this example we are only building a container with Python 3.9 and no other dependencies for simplicity. The `/predictions` folder is our `OUTPUT_DIR` where the returning predictions for this psuedo example will be stored. `evaluator_data` contains 2 sample JSON files, only is a very simple request and the other is more complicated. `evaluator_data` is mounted at run time to increase flexibility.

`evaluator_API_clean_apptainer.py` is the main TCP message passing script that reads in a JSON file and will communicate with the Predictor. It needs to be copied into the container in the `%files` section of the .def file. 

Change the `/path_to/`  in the .def file to the local file path for the script. 

3. Next run: `apptainer build evaluator.sif evaluator_python_only.def` in the `evaluator_container_sample/` folder. This will build the Evaluator container that automatically runs `evaluator_API_clean_apptainer.py`. In this example the Evaluator container only requires 3 arguments in this order: HOST, PORT, OUTPUT_DIR.

4. `evaluator.sif` will be created in the `evaluator_container_sample` folder

### To create the sample Predictor:

1. Download the `predictor_container_sample` folder to create the sample Predictor:

2. The `predictor_python_only.def` will be used to create the Apptainer container for the Predictor. The `/predictor_data` folder is left empty here due to the simplicity of the model but additional files for the model can be placed here. 

`predictor_API_clean_apptainer.py` is the main TCP message passing script that communicates with the Evaluator. It needs to be copied into the container in the `%files` section of the .def file along with the other python files used by the script. The help JSON file is also copied into the container. 

Change the `/path_to/` in the .def file to the local file path for the scripts. 

3. `cd /predictor_container_sample` and next run: `apptainer build predictor.sif predictor_python_only.def`. This will build the Predictor container that automatically runs `predictor_API_clean_apptainer.py`. In this example the Predictor container only requires 2 arguments in this order: HOST, PORT.

4. `predictor.sif` will be created in the `predictor_container_sample` folder

To run the containers:

To get the local host IP you can use `hostname -I`. Ports above 1024 are usually free to use on most computers/servers. 

The predictor needs to be started first. 

1. apptainer run -B /path_to/predictor_data/ predictor.sif HOST PORT
2. apptainer run -B /path_to/evaluator_data:/evaluator_data -B /path/to/predictions:/predictions evaluator.sif HOST PORT OUTPUT_DIR

If the connection was sucssesful a file called `predictor_return_file.json` will be created in the `/path/to/predictions/`

### Helpful hints and notes

+ Note: The `-B` is used to mount local directories for the containers so that they have access to those folders/files. 
+ Note: You many need to change file permissions before building the container if you get permission errors when copying files into the container. This can be done using `chmod 644 file_name` before running the build command.
+ Your data can be stored in any format in the /evaluator_data/ folder. If it's a pre-made JSON format in the API format you can directly using use the evaluator_API_clean_apptainer.py script which reads in JSON file. Otherwise you will have to create the JSON file before you connect to a predictor. 
+ Additional dependencies can be added into the .def files in the `%post section`
