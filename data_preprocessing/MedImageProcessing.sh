#!/bin/bash

########################## MRI and label registration ##########################
#level_flag = 1
N4_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/atlas/
registered_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Neuromorphometric_registration/registration/image/
registered_atlas_label_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Neuromorphometric_registration/registration/label/
mni_image=/home/rtm/scratch/rtm/ms_project/data/icbm/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii
mkdir -p $registered_atlas_path  $registered_atlas_label_path  $N4_atlas_path
########################## CT preprocess ##########################
#level_flag = 2
Threshold_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Threshold/NCCT_ct/
Gassuain_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Gassuain/NCCT_ct/
BET_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/BET/NCCT_ct/
Crop_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/Crop/NCCT_ct/
registered_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/NCCT_ct/
resampled_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/ResampleadImages/NCCT_ct/
template_ct_path=/home/rtm/scratch/rtm/ms_project/data/CT_template/NCCT-template-affine-brain.nii.gz
mkdir -p $Threshold_ct_path $Gassuain_ct_path $BET_ct_path $Crop_ct_path $registered_ct_path $resampled_ct_path

########################## CT Windowing ##########################
#level_flag = 3
lower_bound=-100
upper_bound=1000
Window_ct_path=/home/rtm/scratch/rtm/data/final_dataset/iDB/CT/window_ct/
mni_image=/home/rtm/scratch/rtm/ms_project/data/icbm/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii
mkdir -p $Window_ct_path=

########################## register CT images to MRI ##########################
#level_flag = 4
N4_MRI_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/iDB/N4_MRI/
registered_MRI_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/iDB/Registered_MRI/
registered_CT_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/iDB/Registered_CT/
Window_ct_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/iDB/Thereshold_CT/
mni_image=/home/rtm/scratch/rtm/ms_project/data/icbm/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii
mkdir -p $N4_MRI_path  $registered_MRI_path $registered_CT_path $Window_ct_path

########################## beast skull stripping ##########################
#level_flag = 5
BEAST_PATH=/home/rtm/projects/def-xiaobird/Data/beast
outputDir=/home/rtm/scratch/rtm/data/final_dataset/Neuromorphometrics/SkullStripped/beast
template_PATH=/home/rtm/scratch/rtm/ms_project/data/icbm/mni_icbm152_nlin_sym_09c_minc2
mkdir -p $outputDir






echo 'paths file created'
level_flag=0 # atlas images process
# reading command line arguments
while getopts "a:l:c:w:m:r:b:" OPT
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
    w) #ct_windowing
   ct_image_name=$OPTARG
   level_flag=3
   echo '-w  CT ' $ct_image_name

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

      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #OAS1_0202_MR1.nii

     img_postfix='N4correct_'
	   image_name_N4="$img_postfix$input_basename"
	   full_image_name_N4="$N4_atlas_path$image_name_N4"
	   echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4

      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] \
      -i $image_name -o $full_image_name_N4


      img_postfix='Registered_'
	    image_basename="$img_postfix$input_basename"
	    full_image_name_registered="$registered_atlas_path$image_basename"
	   	echo '$full_image_name_registered ' $full_image_name_registered


	   	img_postfix='omat_'
	   	input_basename_without_postfix=$(echo $input_basename | cut -d . -f 1 -)
	    image_basename="$img_postfix$input_basename_without_postfix.mat"
	    full_image_name_registered_mat="$registered_atlas_path$image_basename"
	   	echo '$full_image_name_registered_mat ' $full_image_name_registered_mat

	   	img_postfix='Registered_'
	   	label_basename="$(basename -- $label_name)"
	    label_basename="$img_postfix$label_basename"
	    full_label_name_registered="$registered_atlas_label_path$label_basename"
	   	echo '$full_label_name_registered ' $full_label_name_registered


      flirt -in $image_name -ref $mni_image -out $full_image_name_registered -omat $full_image_name_registered_mat
      flirt -in $label_name -ref $mni_image -init $full_image_name_registered_mat -applyxfm -out $full_label_name_registered -interp nearestneighbour


fi



########################## CT preprocess ##########################
if [[ $level_flag == 2 ]]; then

      ct_input_basename="$(basename -- $ct_image_name)"
      input_basename_without_postfix=$(echo $ct_input_basename| cut -d . -f 1 -)
      echo 'ct_input_basename' $ct_input_basename

      img_postfix='Threshold_'
	    image_basename="$img_postfix$ct_input_basename"
	    threshold_image_path="$Threshold_ct_path$image_basename"
	   	echo 'threshold_image_path ' $threshold_image_path
	   	ls -l $ct_image_name

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



########################## CT Windowing ##########################
if [[ $level_flag == 3 ]]; then

      ct_input_basename="$(basename -- $ct_image_name)"
      echo 'ct_input_basename' $ct_input_basename

      img_postfix='window_'
	    image_basename="$img_postfix$ct_input_basename"
	    threshold_image_path="$Window_ct_path$image_basename"
	   	echo 'threshold_image_path ' $threshold_image_path
      fslmaths  $ct_image_name -thr $lower_bound -uthr $upper_bound $threshold_image_path
fi


########################## register CT images to MRI ##########################
if [[ $level_flag == 4 ]]; then

      MRI_input_basename="$(basename -- $MRI_image_name)"
      echo 'MRI_input_basename' $MRI_input_basename

      img_postfix='N4correct_'
	    image_name_N4="$img_postfix$MRI_input_basename"
	    full_image_name_N4="$N4_atlas_path$image_name_N4"
	    echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4


      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0]  -i $image_name -o $full_image_name_N4

      img_postfix='Registered_'
	    image_basename="$img_postfix$MRI_input_basename"
	    full_MRI_name_registered="$registered_MRI_path$image_basename"
	   	echo '$full_MRI_name_registered ' $full_MRI_name_registered


	   	img_postfix='omat_'
	   	MRI_input_basename_without_postfix=$(echo $MRI_input_basename | cut -d . -f 1 -)
	    image_basename="$img_postfix$MRI_input_basename_without_postfix.mat"
	    full_MRI_name_registered_mat="$registered_MRI_path$image_basename"
	   	echo '$full_MRI_name_registered_mat ' $full_MRI_name_registered_mat

	   	img_postfix='Registered_'
	   	CT_basename="$(basename -- $CT_image_name)"
	    CT_basename="$img_postfix$CT_basename"
	    full_CT_name_registered="$registered_CT_path$CT_basename"
	   	echo '$full_CT_name_registered ' $full_CT_name_registered


      flirt -in $MRI_image_name -ref $mni_image -out $full_MRI_name_registered -omat $full_MRI_name_registered_mat
      flirt -in $CT_image_name -ref $mni_image -init $full_MRI_name_registered_mat -applyxfm -out $full_CT_name_registered


fi





########################## beast skull stripping ##########################
if [[ $level_flag == 5 ]]; then
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














