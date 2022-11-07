#!/bin/bash

filenames=("atlas" "atlas_label" )
target_image=$1
echo $target_image
prefix='seg_'
segmentation_name=$prefix$target_image
echo $segmentation_name
atlas_txt_path=../data/labelFusion/paths/atlas.txt
atlas_label_txt_path=../data/labelFusion/paths/atlas_label.txt
atlas_txt=`cat $atlas_txt_path`
atlas_label_txt=`cat $atlas_label_txt_path`
save_dir=../data/labelFusion/result
mkdir -p $save_dir

input_atlas_labels_name=" "
while read atlas<&3 && read atlas_label<&4;
do
  input_atlas_labels_name+=" -g ${atlas} -l ${atlas_label}"
done 3<$atlas_txt_path 4<$atlas_label_txt_path


#bash /home/reyhan/programs/ants/ants-2.4.2-ubuntu-22.04-X64-gcc/ants-2.4.2/bin/antsJointLabelFusion.sh \
#  -d 3 \
#  -t /home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/target/sub-0001_space-pet_FLAIR.nii.gz \
#  -o $save_dir/$segmentation_name \
#  -p malfPosteriors%04d.nii.gz \
#  $input_atlas_labels_name

