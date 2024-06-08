#!/bin/bash
# This script extracts ventricle labels from whole brain NIfTI images.
# It uses a Python script (extract_labels.py) and runs it within a Singularity container.

# Load the Python module
module load python

# Prompt the user to enter the path to the Singularity image
echo "Please enter the path to the Singularity image:"
read SINGULARITY_PATH

# Check if the user entered a value
if [ -z "$SINGULARITY_PATH" ]; then
  echo "No Singularity image path entered. Exiting."
  exit 1
fi

# Submit the job using neurogliaSubmit with the specified Singularity image

neurogliaSubmit -I  $SINGULARITY_PATH -j Quick python  ./extract_labels.py