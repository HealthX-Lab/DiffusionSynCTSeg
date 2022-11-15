#!/bin/bash


#target_txt_path=/home/rtm/data/labelFusion/paths/target.txt
target_txt_path=/mnt/data/labelFusion/target/sub-0001_space-pet_FLAIR.nii.gz
bash ./labelfusion.sh "$target_txt_path"

#while read -r line; do
 #   sbatch ./labelfusion.sh "$line"
#done <$target_txt_path