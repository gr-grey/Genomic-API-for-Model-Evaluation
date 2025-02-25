# Configuring Definition File and Running Evaluator Container for DREAM-RNN

## Overview

This container includes:
    - Evaluator API script for genomic sequence evaluation.
    - Installed Python dependencies required by the Evaluator.

## Usage

We encourage using pre-built containers for this model that are hosted on Zenodo: <https://zenodo.org/records/14861069>.

However, if you are building the container using the provided definition file, ensure you have the following directory structure on the host:

```bash
/path/to/DREAM_RNN/src/evaluator_container_apptainer/
├── evaluator.def
├── evaluator.sif
├── evaluator_API_clean_apptainer.py
├── evaluator_data
│   ├── evaluator_input_sample_test.json
│   ├── evaluator_message_more_complex.json
│   └── evaluator_message_simple_test.json
└── predictions
```

### Build the container (SIF)

```bash
apptainer build evaluator.sif evaluator.def
```

### Bind these directories and run the container

```bash
apptainer run \
    -B absolute/path/to/evaluator_data:/evaluator_data \
    -B absolute/path/to/predictions:/predictions \
    evaluator.sif PREDICTOR_HOST PREDICTOR_PORT /path/to/predictions
```

## Details

- The container sends and receives data via a TCP socket and requires `/evaluator_data` and `/predictions` to be mounted at runtime.

## Purpose

- Facilitates genomic sequence evaluation by interacting with the Predictor container.
- Handles input/output through bound directories.

## Example Command

```bash
apptainer run \
    -B absolute/path/to/evaluator_data:/evaluator_data \
    -B absolute/path/to/predictions:/predictions \
    evaluator.sif 172.16.47.244 5000 /predictions
```

## Arguments

1. `PREDICTOR_HOST`: IP address or hostname of the Predictor server.
2. `PREDICTOR_PORT`: Port number the Predictor is listening on.
3. `/predictions`: Path inside the container for storing output predictions.

## Additional Notes about the `%environment` Block in the Definition File

```bash
%environment
    # Prevent automatic binding of host directories
    export APPTAINER_NO_MOUNT="home,tmp,proc,sys,dev"
    export LC_ALL=C
    export PATH="/usr/local/bin:$PATH"
```

- `export APPTAINER_NO_MOUNT="home,tmp,proc,sys,dev"`:
*Why it is required:* By default, Apptainer automatically mounts host directories (like /home directory, /tmp, /proc, /sys, and /dev) into the container. This can inadvertently expose host data or cause conflicts. Setting this variable disables those automatic mounts so that only explicitly bound directories (using the -B flag) will be available inside the container.

- `export LC_ALL=C`:
This sets the container to use the default “C” (POSIX) locale for consistent sorting, formatting, and error messages, regardless of the host’s locale settings.

- `export PATH="/usr/local/bin:$PATH"`:
This prepends /usr/local/bin (as defined within the container) to the PATH, ensuring that container-specific executables in that directory take precedence.

### Additional Links for Reference

- [Apptainer Documentation:](https://apptainer.org/docs/user/latest/)
- [HEP Softwate Foundation -- Introduction to Apptainer/Singularity:](https://hsf-training.github.io/hsf-training-singularity-webpage/)
