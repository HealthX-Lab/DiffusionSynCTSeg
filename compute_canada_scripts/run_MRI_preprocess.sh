#!/bin/bash
#BASE_PATH=/home/rtm/scratch/rtm/ms_project/data/iDB/MNI
BASE_PATH=/home/rtm/scratch/rtm/ms_project/test/iDB
target_mri_path=( $( ls -a ${BASE_PATH}/*T1w.nii.gz))


for i in $(seq ${#target_mri_path[@]}); do
    target_mri=${target_mri_path[i-1]}
    echo $target_mri
    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -t $target_mri

done