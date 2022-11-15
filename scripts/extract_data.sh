##############################################
source_atlas_dir=../data/OASISLabels
new_atlas_dir=../data/labelFusion/Atlas
new_atlas_label_dir=../data/labelFusion/AtlasLabel
target_dataset=../data/iDB-CERMEP/derivatives/coregistration
new_target_images_folder=../data/labelFusion/target

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