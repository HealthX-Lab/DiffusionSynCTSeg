########################## Create a folder for atlas images and fill it ####################
search_dir=../data/OASISLabels
copy_dir=../data/labelFusion/Atlas2
mkdir -p $copy_dir
#function copy_file ()
#{
#  SUB="$3"
#  image="$1"
#  echo $SUB
#  if [[ $image==*"$SUB"*  ]]
#		 then
#		   echo '***'
#		   echo "$1"
#		  	cp "$1" "$2"
#		  	echo '@@@@'
#
#	fi
#	return 1
#}
for folder in "$search_dir"/*
 do
	for image in "$folder"/*
	 do
      SUB='MR1.nii'
		 if [[ $image == *"$SUB"*  ]];
		 then
		   echo $image
		  	cp $image "$copy_dir"

		 fi

		 SUB='MR2.nii'
		 if [[ $image == *"$SUB"*  ]];
		 then
		   echo '###' $image
		  	cp $image "$copy_dir"

		 fi
	 done
done