#!/bin/bash




new_target_images_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/target.txt
new_atlas_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas.txt
SS_target_path=/home/rtm/scratch/rtm/data/labelFusion/SkullStrippedimages/iDB_SS/
SS_atlas_path=/home/rtm/scratch/rtm/data/labelFusion/SkullStrippedimages/atlasOASIS/

inputfoldernames=($new_atlas_folder  $new_target_images_folder)
outputfoldernames=($SS_atlas_path  $SS_target_path)
filenames=("atlas" "target")
mkdir -p $SS_atlas_path $SS_target_path
echo 'paths file created'
for i in 0 1 ; do
    txt_input_name=${inputfoldernames[$i]}
    output_path=${outputfoldernames[$i]}
    while read image_name<&3
    do

      input_basename="$(basename -- $image_name)"
      img_postfix='SkullStripped_'
	   image_basename="$img_postfix$input_basename"
	   full_image_basename="$output_path$image_basename"

	   echo ' $full_image_basename ' $full_image_basename


      neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular  \
      bet $image_name $full_image_basename -m -f 0.15 -g 0.1 -R -B

    done 3<$txt_input_name
done