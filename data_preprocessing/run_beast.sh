#!/bin/bash
BASE_PATH=/home/rtm/scratch/rtm/data/final_dataset/Neuromorphometrics/with_Skull
atlas_path=( $( ls -a ${BASE_PATH}/*.nii.gz))


for i in $(seq ${#atlas_path[@]}); do
    atlas=${atlas_path[i-1]}
    mri_input_basename="$(basename -- $atlas)"
    echo 'mri_input_basename in run' $mri_input_basename
    neurogliaSubmit -I  /project/6055004/tools/singularity/minc-toolkit-1.9.16-min.simg   \
      -j 2core8gb /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -b $atlas


done