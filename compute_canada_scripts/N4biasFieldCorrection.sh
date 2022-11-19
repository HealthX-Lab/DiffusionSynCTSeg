#!/bin/bash




new_target_images_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/target.txt
new_atlas_dir=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas.txt
N4_target_path=/home/rtm/scratch/rtm/data/labelFusion/N4BiasFieldCorrected/target_IBD/
N4_atlas_path=/home/rtm/scratch/rtm/data/labelFusion/N4BiasFieldCorrected/atlas_OASIS/

inputfoldernames=($new_atlas_dir  $new_target_images_folder)
outputfoldernames=($N4_atlas_path  $N4_target_path)
filenames=("atlas" "target")
mkdir -p $N4_atlas_path $N4_target_path
echo 'paths file created'
for i in 0 1 ; do
    txt_input_name=${inputfoldernames[$i]}
    output_path=${outputfoldernames[$i]}
    while read image_name<&3
    do

      input_basename="$(basename -- $image_name)"
	   img_postfix='N4correct_'
	   image_name_N4="$img_postfix$input_basename"
	   full_image_name_N4="$output_path$image_name_N4"
	   echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4


      neurogliaSubmit -I  /project/6055004/tools/singularity/minc-toolkit-1.9.16-min.simg \
      -j Regular  \
      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] \
      -i $image_name -o $full_image_name_N4

    done 3<$txt_input_name
done