new_target_images_folder=../data/labelFusion/target
new_atlas_dir=../data/labelFusion/Atlas
new_atlas_label_dir=../data/labelFusion/AtlasLabel
foldernames=($new_atlas_dir $new_atlas_label_dir $new_target_images_folder)
filenames=("atlas" "atlas_label" "target")
txt_path=../data/labelFusion/paths
mkdir -p $txt_path
for i in 0 1 2; do
    folder_name=${foldernames[$i]}
    txt_name=${filenames[$i]}
    echo $folder_name
    txt_file="$txt_path/$txt_name.txt"
    touch $txt_file
    for image_name in $folder_name/*; do
      echo $image_name
      echo "$image_name\n" >> $txt_file
    done
done


