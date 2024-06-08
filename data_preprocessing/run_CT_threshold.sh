#!/bin/bash

# This script applies a threshold to CT images using a Python script (CT_threshold.py).
# It runs the script within a Singularity container with specified lower and upper bounds.

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

# Submit the job using neurogliaSubmit with the specified Singularity image and threshold values
neurogliaSubmit -I  $SINGULARITY_PATH -j Quick python  ./CT_threshold.py -10 500