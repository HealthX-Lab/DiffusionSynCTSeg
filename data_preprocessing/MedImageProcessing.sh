#!/bin/bash

# This script performs various preprocessing and registration tasks on MRI and CT images.
# It supports MRI and label registration, CT preprocessing, CT registration to MRI, and MRI skull stripping using BEaST.

# Initialize the level_flag variable
level_flag=0
# reading command line arguments
while getopts "a:l:c:m:r:b:" OPT
  do

  case $OPT in
   a) #atlas
    image_name=$OPTARG
    level_flag=1
    echo '-a  atlas' $image_name

   ;;
   l) #label image
   label_name=$OPTARG
   level_flag=1
   echo '-l  label' $label_name
   ;;

   c) #ct scan preprocess
   ct_image_name=$OPTARG
   level_flag=2
   echo '-c  CT' $ct_image_name

   ;;

   m) # CT registration to mri
   MRI_image_name=$OPTARG
   level_flag=4
   echo '-m MRI' $MRI_image_name

   ;;

  r) # CT registration to mri
   CT_image_name=$OPTARG
   level_flag=4
   echo '-r  CT ' $CT_image_name
   ;;


 b) # beast  MRI skull stripping
   inputNII=$OPTARG
   level_flag=5
   echo '-BEaST MRI SS' $inputNII

   ;;

   esac
done

echo 'level_flag' $level_flag

########################## MRI and label registration ##########################
if [[ $level_flag == 1 ]]; then


      N4_atlas_path='path-to-save-N4-mri'
      registered_atlas_path='path-to-save-registered-mri'
      registered_atlas_label_path='path-to-save-registered-labels'
      mni_image='path-to-icbm-template'
      mkdir -p $registered_atlas_path  $registered_atlas_label_path  $N4_atlas_path


      # Extract the basename of the MRI image
      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename

     # Apply N4 Bias Field Correction
     img_postfix='N4correct_'
	   image_name_N4="$img_postfix$input_basename"
	   full_image_name_N4="$N4_atlas_path$image_name_N4"
	   echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4

      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] \
      -i $image_name -o $full_image_name_N4

      # Register MRI to MNI space
      img_postfix='Registered_'
	    image_basename="$img_postfix$input_basename"
	    full_image_name_registered="$registered_atlas_path$image_basename"
	   	echo '$full_image_name_registered ' $full_image_name_registered

      # Generate transformation matrix filename
	   	img_postfix='omat_'
	   	input_basename_without_postfix=$(echo $input_basename | cut -d . -f 1 -)
	    image_basename="$img_postfix$input_basename_without_postfix.mat"
	    full_image_name_registered_mat="$registered_atlas_path$image_basename"
	   	echo '$full_image_name_registered_mat ' $full_image_name_registered_mat

	   	# Register label to MNI space using the same transformation matrix
	   	img_postfix='Registered_'
	   	label_basename="$(basename -- $label_name)"
	    label_basename="$img_postfix$label_basename"
	    full_label_name_registered="$registered_atlas_label_path$label_basename"
	   	echo '$full_label_name_registered ' $full_label_name_registered

      # Perform registration of MRI to MNI space and save the transformation matrix
      flirt -in $image_name -ref $mni_image -out $full_image_name_registered -omat $full_image_name_registered_mat

      # Apply the transformation matrix to register the label to MNI space
      flirt -in $label_name -ref $mni_image -init $full_image_name_registered_mat -applyxfm -out $full_label_name_registered -interp nearestneighbour

      # Histogram matching for MRI image to MNI template
      fslmaths $full_image_name_registered -histmatch $mni_image $full_image_name_registered

fi



########################## CT preprocess ##########################
if [[ $level_flag == 2 ]]; then

      #level_flag = 2
      Threshold_ct_path='path-to-save-threshold-CT'
      Gassuain_ct_path='path-to-save-Gaussian-CT'
      BET_ct_path='path-to-save-skull-stripped-CT'
      Crop_ct_path='path-to-save-brain-cropped-CT'
      registered_ct_path='path-to-save-registered-CT'
      resampled_ct_path='path-to-save-resampled-CT'
      template_ct_path='path-to-CT-template'
      mkdir -p $Threshold_ct_path $Gassuain_ct_path $BET_ct_path $Crop_ct_path $registered_ct_path $resampled_ct_path




      # Extract the basename of the CT image
      ct_input_basename="$(basename -- $ct_image_name)"
      input_basename_without_postfix=$(echo $ct_input_basename| cut -d . -f 1 -)
      echo 'ct_input_basename' $ct_input_basename

      # Apply thresholding to the CT image
      img_postfix='Threshold_'
	    image_basename="$img_postfix$ct_input_basename"
	    threshold_image_path="$Threshold_ct_path$image_basename"
	   	echo 'threshold_image_path ' $threshold_image_path
	   	ls -l $ct_image_name
      fslmaths  $ct_image_name -thr 0 -uthr 100 $threshold_image_path

      # Apply Gaussian blur to the thresholded image
      img_postfix='Gassuain_'
	    image_basename="$img_postfix$ct_input_basename"
	    blur_image_path="$Gassuain_ct_path$image_basename"
	   	echo 'blur_image_path ' $blur_image_path
      fslmaths $threshold_image_path -s 1 $blur_image_path

      # Apply brain extraction (BET) to the blurred image
      img_postfix='SS_'
	    image_basename="$img_postfix$input_basename_without_postfix"
	    SS_image_path="$BET_ct_path$image_basename"
	    SS_image_path_with_nii="$BET_ct_path$img_postfix$ct_input_basename"
	    mask='_mask.nii'
	    bet $blur_image_path $SS_image_path -m -o
	   	echo '$SS_image_path  mask_SS_image_path' $SS_image_path  $mask_SS_image_path
      mask_SS_image_path="$SS_image_path$mask"

      # Apply mask to crop the CT image
      img_postfix='crop_'
	    image_basename="$img_postfix$ct_input_basename"
	    cropped_image_path="$Crop_ct_path$image_basename"
	   	echo 'cropped_image_path   mask_SS_image_path' $cropped_image_path  $mask_SS_image_path
      fslmaths $ct_image_name -mul $mask_SS_image_path $cropped_image_path

      # Register the cropped CT image to the MNI template
      img_postfix='Registered_'
	    image_basename="$img_postfix$ct_input_basename"
	    Registered_image_path="$registered_ct_path$image_basename"

	    # Generate the transformation matrix filename
	    img_postfix='matrix_'
	    matrix_postfix='.mat'
	    image_basename="$img_postfix$input_basename_without_postfix$matrix_postfix"
	    matrix_image_path="$registered_ct_path$image_basename"
	   	echo '$matrix_image_path Registered_image_path  ' $matrix_image_path  $Registered_image_path
	   	flirt -in $cropped_image_path -ref $template_ct_path -out $Registered_image_path -omat $matrix_image_path -cost normmi

      # Resample the CT image based on the transformation matrix
      img_postfix='Resampled_'
	    image_basename="$img_postfix$ct_input_basename"
	    Resampled_image_path="$resampled_ct_path$image_basename"
	   	echo '$Resampled_image_path ' $Resampled_image_path
      flirt -in $ct_image_name -ref $template_ct_path -out $Resampled_image_path -applyxfm -init $matrix_image_path -datatype int

      # Apply histogram matching to the resampled image
      fslmaths $Resampled_image_path -histmatch $template_ct_path $Resampled_image_path

fi



########################## Register CT images to MRI ##########################

# Check if the level_flag is set to 4
if [[ $level_flag == 4 ]]; then

    N4_MRI_path='path-to-save-N4-data'
    registered_MRI_path='path-to-registered-MRI-data'
    registered_CT_path='path-to-registered-CT-data'
    mni_image=/home/rtm/scratch/rtm/ms_project/data/icbm/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii
    mkdir -p $N4_MRI_path  $registered_MRI_path $registered_CT_path

    # Extract the basename of the MRI image
    MRI_input_basename="$(basename -- $MRI_image_name)"
    echo 'MRI_input_basename: ' $MRI_input_basename

    # N4 Bias Field Correction
    img_postfix='N4correct_'
    image_name_N4="$img_postfix$MRI_input_basename"
    full_image_name_N4="$N4_atlas_path$image_name_N4"
    echo 'MRI image: ' $MRI_image_name ' | N4 corrected image: ' $full_image_name_N4

    # Apply N4BiasFieldCorrection
    N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] -i $MRI_image_name -o $full_image_name_N4

    # Register MRI to MNI space
    img_postfix='Registered_'
    image_basename="$img_postfix$MRI_input_basename"
    full_MRI_name_registered="$registered_MRI_path$image_basename"
    echo 'Registered MRI: ' $full_MRI_name_registered

    # Generate transformation matrix filename
    img_postfix='omat_'
    MRI_input_basename_without_postfix=$(echo $MRI_input_basename | cut -d . -f 1 -)
    image_basename="$img_postfix$MRI_input_basename_without_postfix.mat"
    full_MRI_name_registered_mat="$registered_MRI_path$image_basename"
    echo 'Transformation matrix: ' $full_MRI_name_registered_mat

    # Register CT to MNI space using the same transformation matrix
    img_postfix='Registered_'
    CT_basename="$(basename -- $CT_image_name)"
    CT_basename="$img_postfix$CT_basename"
    full_CT_name_registered="$registered_CT_path$CT_basename"
    echo 'Registered CT: ' $full_CT_name_registered

    # Perform registration of MRI to MNI space and save the transformation matrix
    flirt -in $MRI_image_name -ref $mni_image -out $full_MRI_name_registered -omat $full_MRI_name_registered_mat

    # Apply the transformation matrix to register the CT image to MNI space
    flirt -in $CT_image_name -ref $mni_image -init $full_MRI_name_registered_mat -applyxfm -out $full_CT_name_registered

    # Histogram matching for MRI and CT images to MNI template
    fslmaths $full_MRI_name_registered -histmatch $mni_image $full_MRI_name_registered
    fslmaths $full_CT_name_registered -histmatch $mni_image $full_CT_name_registered

fi




########################## beast skull stripping ##########################
if [[ $level_flag == 5 ]]; then
      BEAST_PATH='path-to-mri-data'
      outputDir='output-folder'
      template_PATH='path-to-the-ICBM-template'
      mkdir -p $outputDir


      set -e

      mri_input_basename="$(basename -- $inputNII)"
      echo 'mri_input_basename' $mri_input_basename #sub-0027_space-pet_FLAIR.nii.gz
      basename=$(echo $mri_input_basename | cut -d . -f 1 -)

      TDIR=$(mktemp -d -p /home/rtm/scratch/rtm/data/MedImagepreprocess/beast)
      echo 'TDIR' $TDIR
      trap "{ cd - ; rm -rf $TDIR; exit 255; }" SIGINT

      input=$TDIR/${basename}_minc.mnc
      nu_1=$TDIR/${basename}_nu_1.mnc
      nu_2=$TDIR/${basename}_nu_2.mnc
      head_mni=$TDIR/${basename}_head_mni.mnc
      mask_mni=$TDIR/${basename}_mask_mni.mnc
      toTalxfm_1=$TDIR/${basename}_toTal.xfm
      toTalxfm_2=$outputDir/${basename}_stx.xfm
      mask=$TDIR/${basename}_mask_native.mnc
      brain=$TDIR/${basename}_brain_native.mnc
      final=$TDIR/${basename}_final.mnc
      final_2=$TDIR/${basename}_final2.mnc
      final_3=$outputDir/${basename}_final3.mnc
      norm=$outputDir/${basename}_stx.mnc
      pik=$outputDir/${basename}_pik.png
      mask_template=$outputDir/${basename}_mask_stx.mnc
      final_image_nii=$outputDir/${basename}_Skull_Stripped.nii
      before_reg_image_nii=$outputDir/${basename}_Skull_Stripped_before_reg.nii




      # 0. nifty to minc conversion
      nii2mnc $inputNII $input -unsigned -short

      # 1. rough non-uniformity correction
      N4BiasFieldCorrection -d 3 -b [200] -c [200x200x200x200,0.0] -i $input -o $nu_1

      # 2.1 BEast skull stripping while putting the brain to the MNI space
      beast_normalize $nu_1 $head_mni $toTalxfm_1 -modeldir $template_PATH
      mincbeast -fill -median -conf $BEAST_PATH/default.1mm.conf $BEAST_PATH $head_mni $mask_mni

      # 2.2 resample the mask back to the native space
      mincresample $mask_mni -like $input -invert_transformation -transform $toTalxfm_1 $mask -short -nearest

      # 2.3 remove the skull in the native space
      minccalc -expr "A[0]>0.5?A[1]:0" $mask $input $brain -short -unsigned

      # 3. refined N4 non-uniformity correction
      N4BiasFieldCorrection -d 3 --verbose -r 1 -x $mask -b [200] -c [300x300x300x200,0.0] -i $brain -o $nu_2 --histogram-sharpening [0.05,0.01,1000]

      mnc2nii  $nu_2  $before_reg_image_nii

      # 4.1 Linear registration to target template ICBM152 or ADNI local
      bestlinreg_s -lsq12 -nmi -source_mask $mask -target_mask $template_PATH/mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc $nu_2 $template_PATH/mni_icbm152_t1_tal_nlin_sym_09c.mnc $toTalxfm_2 -clobber
      #bestlinreg.pl -lsq12 -source_mask $mask -target_mask $MNItemplatePATH/mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc $nu_2 $MNItemplatePATH/mni_icbm152_t1_tal_nlin_sym_09c.mnc $toTalxfm_2 -clobber

      # 4.2 resample the image to ICBM sapce with 12 param registration
      itk_resample --short --transform $toTalxfm_2 --like $template_PATH/mni_icbm152_t1_tal_nlin_sym_09c.mnc $nu_2 $final --clobber
      mincresample -short -transform $toTalxfm_2 -like $template_PATH/mni_icbm152_t1_tal_nlin_sym_09c.mnc $nu_2 $final_2 -trilinear -clobber
      minccalc -expr "A[0]<0?A[1]:A[0]" $final $final_2 $final_3 -short -signed -clobber

      # Convert image from MNC to NII format.
      mnc2nii  $final_3  $final_image_nii



      #5. intensity normalization
      mincresample $mask -like $template_PATH/mni_icbm152_t1_tal_nlin_sym_09c.mnc -transform $toTalxfm_2 $mask_template -short -nearest
      volume_pol --verbose --clobber --order 1 --noclamp --source_mask $mask_template --target_mask $template_PATH/mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc $final_3 $template_PATH/mni_icbm152_t1_tal_nlin_sym_09c.mnc $norm

      #6. quality check
      mincpik --scale 2 $norm --slice 60 -z $pik -clobber

#      cd -
      cd /home/rtm/scratch/rtm/ms_project/domain_adaptation_CTscan/scripts
      rm -rf $TDIR
fi
















