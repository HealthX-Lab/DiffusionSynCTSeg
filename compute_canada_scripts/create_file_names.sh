#!/bin/bash
new_target_images_folder=/home/rtm/scratch/rtm/data/labelFusion/target
new_atlas_dir=/home/rtm/scratch/rtm/data/labelFusion/Atlas
#!/bin/bash
new_target_images_folder=/home/rtm/scratch/rtm/data/labelFusion/target
new_atlas_dir=/home/rtm/scratch/rtm/data/labelFusion/Atlas
new_atlas_label_dir=/home/rtm/scratch/rtm/data/labelFusion/AtlasLabel
foldernames=($new_atlas_dir $new_atlas_label_dir $new_target_images_folder)
filenames=("atlas" "atlas_label" "target")
txt_path=/home/rtm/scratch/rtm/data/labelFusion/paths
mkdir -p $txt_path
echo 'paths file created'
for i in 0 1 2; do
    folder_name=${foldernames[$i]}
    txt_name=${filenames[$i]}
    echo $folder_name
    txt_file="$txt_path/$txt_name.txt"
    touch $txt_file
    for image_name in $folder_name/*; do
      echo $image_name
      echo '**********'
      echo "$image_name" >> $txt_file
    done
done



