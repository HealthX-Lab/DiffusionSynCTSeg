#!/bin/bash

filenames=("atlas" "atlas_label" )
target_image=$1
echo $target_image
prefix='malfPosteriors'
imagename="$(basename -- $target_image )"
segmentation_name="$prefix$imagename"
echo '****####' $segmentation_name

atlas_txt_path=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas_mnt.txt
atlas_label_txt_path=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas_label_mnt.txt
atlas_txt=`cat $atlas_txt_path`
atlas_label_txt=`cat $atlas_label_txt_path`
save_dir=/mnt/data/labelFusion/result
#mkdir -p $save_dir

input_atlas_labels_name=" "
while read atlas<&3 && read atlas_label<&4;
do
  input_atlas_labels_name+=" -g ${atlas} -l ${atlas_label}  "
done 3<$atlas_txt_path 4<$atlas_label_txt_path

echo $input_atlas_labels_name

IMAGE_DIR=$save_dir/$segmentation_name/
mkdir -p /home/rtm/scratch/rtm/data/labelFusion/result/$segmentation_name/

singularity exec --env ANTSPATH=/opt/minc/1.9.16/bin/ --bind /home/rtm/scratch/rtm:/mnt /home/rtm/projects/def-xiaobird/tools/singularity/minc-toolkit-1.9.16-min.simg /bin/bash antsJointLabelFusion.sh \
  -d 3 \
  -t $target_image \
  -o $IMAGE_DIR \
  -p $segmentation_name \
  $input_atlas_labels_name

