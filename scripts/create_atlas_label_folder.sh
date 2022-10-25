SUB='seg'
search_dir=/home/reyhan/Projects/domain_adaptation_CTscan/data/OASISLabels
copy_dir=/home/reyhan/Projects/domain_adaptation_CTscan/data/labelFusion/AtlasLabel
for folder in "$search_dir"/*
 do
	for image in "$folder"/*
	 do

		 if [[ $image == *"$SUB"*  ]];
		 then
		   echo $image
		  	cp $image "$copy_dir"
		  fi
		done
done