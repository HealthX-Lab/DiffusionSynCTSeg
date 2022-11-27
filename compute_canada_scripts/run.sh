#!/bin/bash






#BASE_PATH=/home/rtm/scratch/rtm/ms_project/data/projectNeuromorphometrics/ADNILabels
BASE_PATH=/home/rtm/scratch/rtm/ms_project/test/ADNILabels
atlas_path=( $( ls -a ${BASE_PATH}/ADNI*.nii))
label_path=( $( ls -a ${BASE_PATH}/*seg.nii))


for i in $(seq ${#atlas_path[@]}); do
    atlas=${atlas_path[i-1]}
    label=${label_path[i-1]}
    echo $atlas $label
    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -a $atlas -l $label
#
#    input_atlas_labels_name+=" -g $atlas -l $label"

done

echo $input_atlas_labels_name

#BASE_PATH=/home/rtm/scratch/rtm/ms_project/data/projectNeuromorphometrics/OASISLabels
BASE_PATH=/home/rtm/scratch/rtm/ms_project/test/OASISLabels
atlas_path=( $( ls -a ${BASE_PATH}/*_MR1.nii))
label_path=( $( ls -a ${BASE_PATH}/*seg.nii))


for i in $(seq ${#atlas_path[@]}); do
    atlas=${atlas_path[i-1]}
    label=${label_path[i-1]}
    echo $atlas $label
    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -a $atlas -l $label

done

#BASE_PATH=/home/rtm/scratch/rtm/ms_project/data/iDB/MNI
BASE_PATH=/home/rtm/scratch/rtm/ms_project/test/iDB
target_mri_path=( $( ls -a ${BASE_PATH}/*T1w.nii.gz))


for i in $(seq ${#target_mri_path[@]}); do
    target_mri=${target_mri_path[i-1]}
    echo $target_mri
    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -t $target_mri

done


#BASE_PATH=/home/rtm/scratch/rtm/ms_project/data/iDB/MNI
BASE_PATH=/home/rtm/scratch/rtm/ms_project/test/iDB
target_mri_path=( $( ls -a ${BASE_PATH}/*T1w.nii.gz))
target_ct_path=( $( ls -a ${BASE_PATH}/*ct.nii.gz))


for i in $(seq ${#target_mri_path[@]}); do
    target_mri=${target_mri_path[i-1]}
    target_ct=${target_ct_path[i-1]}

    echo $target_mri
    neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -c $target_ct -m $target_mri

done

























