###############################################
source_atlas_dir=../data/OASISLabels
new_atlas_dir=../data/labelFusion/Atlas
new_atlas_label_dir=../data/labelFusion/AtlasLabel
target_dataset=../data/iDB-CERMEP/derivatives/coregistration
new_target_images_folder=../data/labelFusion/target
registered_image_dir=../data/labelFusion/iDB-Registered
result_path=../data/labelFusion/result
#################################################
#Create a folder for atlas images and copy images from  source_atlas_dir to the new_atlas_dir
#
mkdir -p $new_atlas_dir

for folder in "$source_atlas_dir"/*
 do
	for image in "$folder"/*
	 do
      SUB='MR1.nii'
		 if [[ $image == *"$SUB"*  ]];
		 then
		  	cp $image "$new_atlas_dir"

		 fi

		 SUB='MR2.nii'
		 if [[ $image == *"$SUB"*  ]];
		 then
		  	cp $image "$new_atlas_dir"

		 fi
	 done
done
echo 'coping source images to the Atlas folder completed'
#################################################
#Create a folder for atlas labels and copy images from  source_atlas_dir to the new_atlas_dir

SUB='seg'

mkdir -p $new_atlas_label_dir
for folder in "$source_atlas_dir"/*
 do
	for image in "$folder"/*
	 do

		 if [[ $image == *"$SUB"*  ]];
		 then
		  	cp $image "$new_atlas_label_dir"
		  fi
		done
done
echo 'coping source labels to the atlasLabel folder completed'
################################################
# Create a folder for target images that wants to segment and copy images from specific dataset (target_dataset)to new_target_images_folder
SUB='FLAIR'

mkdir -p $new_target_images_folder

for folder in "$target_dataset"/*
 do
	for image in "$folder"/*
	 do

		 if [[ $image == *"$SUB"*  ]];
		 then
		  	cp $image "$new_target_images_folder"

		 fi

	 done
done
echo 'coping target data to the target folder completed'
################################################
# Create a folder for registered images (iDB-registered2) and fill it with registered images

mkdir -p $registered_image_dir

for target_image in "$new_target_images_folder"/*
 do

	   target_basename="$(basename -- $target_image)"
	   target_basename=${target_basename%.nii.gz}
	   full_path="$registered_image_dir/$target_basename"
	   mkdir -p $full_path

	   dir=$(pwd)
	   parentdir="$(dirname "$dir")"
     target_image="$parentdir$target_image"
     target_image=${target_image//..}

	   i=0
	   for atlas_image in "$new_atlas_dir"/*
	   do
      atlas_image="$parentdir$atlas_image"
      atlas_image=${atlas_image//..}
	     label_prefix='_seg.nii'
       label_basename="$(basename -- $atlas_image)"
       label_basename=${label_basename%.nii}
       label_basename="$label_basename$label_prefix"
      label_full_path="$new_atlas_label_dir/$label_basename"
      label_full_path="$parentdir$label_full_path"
      label_full_path=${label_full_path//..}

	     i=$((i+1))
      cd $full_path
	     ANTS 3 -m CC[$target_image,$atlas_image,1,2] -i 10x50x50x20 -o sub"$i"-to-t1 -t SyN[0.25] -r Gauss[3,0]
       WarpImageMultiTransform 3 $label_full_path Sub"$i"-to-t1-ventricle-lin.nii.gz -R $target_image sub"$i"-to-t1Warp.nii.gz sub"$i"-to-t1Affine.txt
	    cd -
	   done

done
echo 'image registration completed. Result are saved in registered folder'
###############################################
# Average segmented labels and save the result in the result folder


mkdir -p $result_path

for image_folder in "$registered_image_dir"/*
 do
     i=0
     declare -a ventricle_segmentation
	   for segmentation_label in "$image_folder"/*
	   do
	       SUB='ventricle'
         if [[ $segmentation_label == *"$SUB"*  ]];
         then
           ventricle_segmentation[i]=$segmentation_label
           i=$((i+1))

          label_prefix='_segmentation.nii'
          label_basename="$(basename -- $image_folder )"
          label_basename="$label_basename$label_prefix"
          label_full_path="$result_path/$label_basename"

         fi
	     done
	     AverageImages 3 $label_full_path 0 ${ventricle_segmentation[@]}

done

echo 'Averaging completed. Results are saved in the result folder.'