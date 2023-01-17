#!/bin/bash
BASE_MRI_PATH=/home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/iBD
BASE_CT_PATH=/home/rtm/scratch/rtm/data/MedImagepreprocess/iDB_delete_nan/ct
MRI_path=( $( ls -a ${BASE_MRI_PATH}/*T1w.nii.gz ))
CT_path=( $( ls -a ${BASE_CT_PATH}/*ct.nii.gz))



for i in $(seq ${#MRI_path[@]}); do
    MRI=${MRI_path[i-1]}
    CT=${CT_path[i-1]}
    echo 'MRI CT paths: '$MRI $CT '**********'

    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Quick /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -m  $MRI -r $CT

done