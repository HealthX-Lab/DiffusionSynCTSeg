#SUB='FLAIR'

segmented_image_dir=/home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/iDB-Segmentation
result_path=/home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/result



for image_folder in "$segmented_image_dir"/*
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
#       AverageImages 3 $label_full_path 0 ventricle_segmentation[0] ventricle_segmentation[1]
done