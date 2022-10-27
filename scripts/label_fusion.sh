#SUB='FLAIR'
atlas_dir=/home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/Atlas
label_atlas_dir=/home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/AtlasLabel
target_image_dir=/home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/target
segmented_image_dir=/home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/iDB-Segmentation


for target_image in "$target_image_dir"/*
 do

	   target_basename="$(basename -- $target_image)"
	   target_basename=${target_basename%.nii.gz}
	   full_path="$segmented_image_dir/$target_basename"
	   mkdir $full_path
	   cd $full_path
	   i=0
	   for atlas_image in "$atlas_dir"/*
	   do
	     label_prefix='_seg.nii'
       label_basename="$(basename -- $atlas_image)"
       label_basename=${label_basename%.nii}
       label_basename="$label_basename$label_prefix"
      label_full_path="$label_atlas_dir/$label_basename"
      echo label_full_path
	     echo $label_full_path
	     i=$((i+1))
	     ANTS 3 -m CC[$target_image,$atlas_image,1,2] -i 10x50x50x20 -o sub"$i"-to-t1 -t SyN[0.25] -r Gauss[3,0]
       WarpImageMultiTransform 3 $label_full_path Sub"$i"-to-t1-ventricle-lin.nii.gz -R $target_image sub"$i"-to-t1Warp.nii.gz sub"$i"-to-t1Affine.txt
	   done
done