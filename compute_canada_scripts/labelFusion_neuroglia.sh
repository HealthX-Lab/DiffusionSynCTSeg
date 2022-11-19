#!/bin/bash

export ANTSPATH=/opt/minc/1.9.16/bin/

atlas_txt_path=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas.txt
atlas_label_txt_path=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas_label.txt
target_txt_path=/home/rtm/scratch/rtm/data/labelFusion/paths/target.txt
save_dir=/home/rtm/scratch/rtm/data/labelFusion/result/

input_atlas_labels_name=" "
while read atlas<&3 && read atlas_label<&4;
do
  input_atlas_labels_name+=" -g ${atlas} -l ${atlas_label}  "

done 3<$atlas_txt_path 4<$atlas_label_txt_path

while read -r target; do

  target_basename="$(basename -- $target)"
  echo 'target_basename' $target_basename #sub-0001_space-pet_FLAIR.nii.gz
  target_basename_without_postfix=$(echo $target_basename | cut -d . -f 1 -)
  echo 'target_basename_without_postfix' $target_basename_without_postfix  #sub-0001_space-pet_FLAIR
  output_path="$save_dir$target_basename_without_postfix/"
  echo 'output_path' $output_path #/home/rtm/scratch/rtm/data/labelFusion/result/sub-0001_space-pet_FLAIR/

  mkdir $output_path

  neurogliaSubmit -I /project/6055004/tools/singularity/minc-toolkit-1.9.16-min.simg -j Regular /scratch/rtm/rtm/ms_project/domain_adaptation_CTscan/scripts/antsJointLabelFusion.sh -d 3 \
  -t $target \
  -o output_path \
  -p malfPosteriors%04d.nii.gz \
  $input_atlas_labels_name

done <$target_txt_path





