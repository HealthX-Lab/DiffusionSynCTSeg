#!/bin/bash
filenames=("atlas" "atlas_label" "target")
filenames_mnt=("atlas_mnt" "atlas_label_mnt" "target_mnt")
txt_path=/home/rtm/scratch/rtm/data/labelFusion/paths
firstString='/home/rtm/scratch/rtm/'
secondString='/mnt/'
mkdir -p $txt_path
echo 'paths file created'
for i in 0 1 2; do
    txt_name=${filenames[$i]}
    mnt_txt_name=${filenames_mnt[$i]}
    txt_file="$txt_path/$txt_name.txt"
    mnt_txt_file="$txt_path/$mnt_txt_name.txt"
    touch $mnt_txt_file
    while read -r line; do
        echo "${line/"$firstString"/"$secondString"}"  >> $mnt_txt_file
    done <$txt_file
done

