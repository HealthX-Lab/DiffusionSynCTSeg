#!/bin/bash

#########################ADNILabels#########################

BASE_PATH=/home/rtm/scratch/rtm/ms_project/data/projectNeuromorphometrics/ADNILabels
atlas_path=( $( ls -a ${BASE_PATH}/*/ADNI*.nii))
label_path=( $( ls -a ${BASE_PATH}/*/*seg.nii))

#########################OASISLabels########################
#BASE_PATH=/home/rtm/scratch/rtm/ms_project/data/projectNeuromorphometrics/OASISLabels
#atlas_path=( $( ls -a ${BASE_PATH}/*/*_MR1.nii and ls -a ${BASE_PATH}/*/*_MR2.nii ))
#label_path=( $( ls -a ${BASE_PATH}/*/*seg.nii))

for i in $(seq ${#atlas_path[@]}); do
    atlas=${atlas_path[i-1]}
    label=${label_path[i-1]}
    echo $atlas $label
    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -a $atlas -l $label


done

echo $input_atlas_labels_name

