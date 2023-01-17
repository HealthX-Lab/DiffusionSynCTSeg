#!/bin/bash

BASE_PATH=/home/rtm/scratch/rtm/ms_project/test/iDB
target_ct_path=( $( ls -a ${BASE_PATH}/*ct.nii.gz))


for i in $(seq ${#target_ct_path[@]}); do
    target_ct=${target_ct_path[i-1]}

    echo $target_mri
    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh   -c $target_ct

done
