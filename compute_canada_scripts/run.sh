#!/bin/bash


new_target_images_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/target.txt
new_atlas_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas.txt
new_atlas_label_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas_label.txt



while read image_name<&3 && read label_name<&4
    do

      neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -a $image_name -l $label_name


done 3<$new_atlas_folder  4<$new_atlas_label_folder




while read image_name<&3
    do
      neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts/MedImageProcessing.sh  -t $image_name $label_name

done 3<$new_target_images_folder






















