#!/bin/bash

###### mri paths
N4_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/atlas/
N4_target_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/iBD/
SS_target_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/iDB_mri/
SS_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/atlas/
registered_target_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/iDB_mri/
registered_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/atlas/
registered_atlas_label_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/atlas_label/
mni_image=/home/rtm/scratch/rtm/ms_project/data/icbm/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii
mni_GT=/home/rtm/scratch/rtm/data/icbm/mni_icbm152_t1_lateralventricles.nii.gz

###### CT paths
Threshold_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Threshold/iDB_ct/
Gassuain_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Gassuain/iDB_ct/
BET_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/BET/iDB_ct/
Crop_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Crop/iDB_ct/
registered_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/iDB_ct/
resampled_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/ResampleadImages/iDB_ct/
template_ct_path=/home/rtm/scratch/rtm/ms_project/data/CT_template/NCCT-template-affine-brain.nii.gz


mkdir -p $Threshold_ct_path $Gassuain_ct_path $BET_ct_path $Crop_ct_path $registered_ct_path $resampled_ct_path
mkdir -p $registered_atlas_path $registered_target_path $registered_atlas_label_path $SS_target_path $SS_atlas_path $N4_target_path $N4_atlas_path
echo 'paths file created'
level_flag=0 # atlas images process
# reading command line arguments
while getopts "a:l:t:c:" OPT
  do
  case $OPT in
   a) #atlas
    image_name=$OPTARG
    echo '-a  atlas' $image_name

   ;;
   l) #label image
   label_name=$OPTARG
   echo '-l  label' $label_name
   ;;


   t) #targetimage
   image_name=$OPTARG
   level_flag=1
   echo '-t  target MRI' $image_name

   ;;
   c) #ct_target
   ct_image_name=$OPTARG
   level_flag=2
   echo '-c  target CT' $ct_image_name

   ;;

   esac
done

echo 'level_flag' $level_flag


###### preprocessed ADNI and OASIS
if [[ $level_flag == 0 ]]; then

      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #OAS1_0202_MR1.nii

      img_postfix='N4correct_'
	   image_name_N4="$img_postfix$input_basename"
	   full_image_name_N4="$N4_atlas_path$image_name_N4"
	   echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4
	   #$image_name  /home/rtm/scratch/rtm/data/labelFusion/Atlas/OAS1_0202_MR1.nii  $full_image_name_N4  /home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/atlas_OASIS2/N4correct_OAS1_0202_MR1.nii


      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] \
      -i $image_name -o $full_image_name_N4


    img_postfix='Registered_'
	   image_basename="$img_postfix$input_basename"
	   full_image_name_registered="$registered_atlas_path$image_basename"
	   	echo '$full_image_name_registered ' $full_image_name_registered
	   	#$full_image_name_registered  /home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/OASIS_atlas_registered2/Registered_OAS1_0202_MR1.nii

	   	img_postfix='omat_'
	   	input_basename_without_postfix=$(echo $input_basename | cut -d . -f 1 -)
	   image_basename="$img_postfix$input_basename_without_postfix.mat"
	   full_image_name_registered_mat="$registered_atlas_path$image_basename"
	   	echo '$full_image_name_registered_mat ' $full_image_name_registered_mat
	   	#$full_image_name_registered_mat  /home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/OASIS_atlas_registered2/omat_OAS1_0249_MR2.mat

	   	img_postfix='Registered_'
	   	label_basename="$(basename -- $label_name)"
	   label_basename="$img_postfix$label_basename"
	   full_label_name_registered="$registered_atlas_label_path$label_basename"
	   	echo '$full_label_name_registered ' $full_label_name_registered
	   	#$full_image_name_registered  /home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/OASIS_atlas_label_registered2/Registered_OAS1_0249_MR2_seg.nii


      flirt -in $full_image_name_N4 -ref $mni_image -out $full_image_name_registered -omat $full_image_name_registered_mat
      flirt -in $label_name -ref $mni_image -init $full_image_name_registered_mat -applyxfm -out $full_label_name_registered -interp nearestneighbour

      img_postfix='SkullStripped_'
	   image_basename="$img_postfix$input_basename_without_postfix"
	   full_image_name_SS="$SS_atlas_path$image_basename"
	   echo '$full_image_name_SS ' $full_image_name_SS
	   #$full_image_name_SS  /home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/atlasOASIS2/SkullStripped_OAS1_0202_MR1.nii

      bet $full_image_name_registered $full_image_name_SS -m -f 0.15 -g 0.1 -R -B

fi



###### preprocessed mni_iDB
if [[ $level_flag == 1 ]]; then


      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #sub-0027_space-pet_FLAIR.nii.gz

      img_postfix='N4correct_'
	   image_name_N4="$img_postfix$input_basename"
	   full_image_name_N4="$N4_target_path$image_name_N4"
	   echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4
	   #$image_name  /home/rtm/scratch/rtm/data/labelFusion/target/sub-0027_space-pet_FLAIR.nii.gz  $full_image_name_N4  /home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/target_IBD2/N4correct_sub-0027_space-pet_FLAIR.nii.gz

      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] \
      -i $image_name -o $full_image_name_N4


     img_postfix='SkullStripped_'
     input_basename_without_postfix=$(echo $input_basename | cut -d . -f 1 -)
	   image_basename="$img_postfix$input_basename_without_postfix"
	   full_image_name_SS="$SS_target_path$image_basename"

      bet $full_image_name_N4 $full_image_name_SS -m -f 0.15 -g 0.1 -R -B

fi

###### registered mni_iDB CT to MRI
if [[ $level_flag == 2 ]]; then

      ct_input_basename="$(basename -- $ct_image_name)"
      input_basename_without_postfix=$(echo $ct_input_basename| cut -d . -f 1 -)
      echo 'ct_input_basename' $ct_input_basename #sub-0027_space-pet_FLAIR.nii.gz

      img_postfix='Threshold_'
	    image_basename="$img_postfix$ct_input_basename"
	    threshold_image_path="$Threshold_ct_path$image_basename"
	   	echo 'threshold_image_path ' $threshold_image_path
	   	ls -l $ct_image_name

#      fslmaths -dt int $ct_image_name -thr 0 -uthr 100 $threshold_image_path
      fslmaths  $ct_image_name -thr 0 -uthr 100 $threshold_image_path



      img_postfix='Gassuain_'
	    image_basename="$img_postfix$ct_input_basename"
	    blur_image_path="$Gassuain_ct_path$image_basename"
	   	echo 'blur_image_path ' $blur_image_path
      fslmaths $threshold_image_path -s 1 $blur_image_path


      img_postfix='SS_'
	    image_basename="$img_postfix$input_basename_without_postfix"
	    SS_image_path="$BET_ct_path$image_basename"
	    SS_image_path_with_nii="$BET_ct_path$img_postfix$ct_input_basename"
	    mask='_mask.nii'
	    bet $blur_image_path $SS_image_path -m -o
	   	echo '$SS_image_path  mask_SS_image_path' $SS_image_path  $mask_SS_image_path
      mask_SS_image_path="$SS_image_path$mask"

      img_postfix='crop_'
	    image_basename="$img_postfix$ct_input_basename"
	    cropped_image_path="$Crop_ct_path$image_basename"
	   	echo 'cropped_image_path   mask_SS_image_path' $cropped_image_path  $mask_SS_image_path
      fslmaths $ct_image_name -mul $mask_SS_image_path $cropped_image_path

      img_postfix='Registered_'
	    image_basename="$img_postfix$ct_input_basename"
	    Registered_image_path="$registered_ct_path$image_basename"

	    img_postfix='matrix_'
	    matrix_postfix='.mat'
	    image_basename="$img_postfix$input_basename_without_postfix$matrix_postfix"
	    matrix_image_path="$registered_ct_path$image_basename"
	   	echo '$matrix_image_path Registered_image_path  ' $matrix_image_path  $Registered_image_path
	   	flirt -in $cropped_image_path -ref $template_ct_path -out $Registered_image_path -omat $matrix_image_path -cost normmi

      img_postfix='Resampled_'
	    image_basename="$img_postfix$ct_input_basename"
	    Resampled_image_path="$resampled_ct_path$image_basename"
	   	echo '$Resampled_image_path ' $Resampled_image_path
      flirt -in $ct_image_name -ref $template_ct_path -out $Resampled_image_path -applyxfm -init $matrix_image_path -datatype int




fi















