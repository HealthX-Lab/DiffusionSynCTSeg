#!/bin/bash


N4_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/atlas_OASIS2/
N4_target_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/N4BiasFieldCorrected/target_IBD2/

SS_target_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/iDB_SS2/
SS_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/atlasOASIS2/


registered_target_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/iDB_target_registered2/
registered_atlas_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/OASIS_atlas_registered2/
registered_atlas_label_path=/home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/OASIS_atlas_label_registered2/

mni_image=/home/rtm/scratch/rtm/data/icbm/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz
mni_GT=/home/rtm/scratch/rtm/data/icbm/mni_icbm152_t1_lateralventricles.nii.gz


mkdir -p $registered_atlas_path $registered_target_path $registered_atlas_label_path $SS_target_path $SS_atlas_path $N4_target_path $N4_atlas_path
echo 'paths file created'
level_flag=0 # atlas images process
# reading command line arguments
while getopts "a:l:t:" OPT
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
   echo '-t  target' $image_name

   ;;
   esac
done

echo 'level_flag' $level_flag



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




      img_postfix='SkullStripped_'
	   image_basename="$img_postfix$input_basename"
	   full_image_name_SS="$SS_atlas_path$image_basename"
	   echo '$full_image_name_SS ' $full_image_name_SS
	   #$full_image_name_SS  /home/rtm/scratch/rtm/data/MedImagepreprocess/SkullStrippedimages/atlasOASIS2/SkullStripped_OAS1_0202_MR1.nii

      bet $full_image_name_N4 $full_image_name_SS -m -f 0.15 -g 0.1 -R -B



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


      flirt -in $full_image_name_SS -ref $mni_image -out $full_image_name_registered -omat full_image_name_registered_mat
      flirt -in $label_name -ref $mni_GT -init full_image_name_registered_mat -applyxfm -out $full_label_name_registered -interp nearestneighbour

    ###############################################3

fi




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
	   image_basename="$img_postfix$input_basename"
	   full_image_name_SS="$SS_target_path$image_basename"

      bet $full_image_name_N4 $full_image_name_SS -m -f 0.15 -g 0.1 -R -B


      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #sub-0001_space-pet_FLAIR.nii.gz
      input_basename_without_postfix=$(echo $input_basename | cut -d . -f 1 -)
      echo 'input_basename_without_postfix' $input_basename_without_postfix  #sub-0027_space-pet_FLAIR


      img_postfix='template_'
      template_path="$registered_target_path$img_postfix$input_basename_without_postfix"
      echo 'template_path' $template_path # template_path /home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/iDB_target_registered2/template_sub-0027_space-pet_FLAIR

      postfixWrap='Wrap.nii.gz'
      template_wrapped_path="$template_path$postfixWrap"
      echo 'template_wrapped_path' $template_wrapped_path #template_wrapped_path /home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/iDB_target_registered2/template_sub-0027_space-pet_FLAIRWrap.nii.gz

      postfixAffine='Affine.txt'
      template_Affine_path="$template_path$postfixAffine"
      echo 'template_Affine_path' $template_Affine_path #template_Affine_path /home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/iDB_target_registered2/template_sub-0027_space-pet_FLAIRAffine.txt

      img_postfix='Wrap_'
	   image_basename="$img_postfix$input_basename"
	   Wrapped_image_path="$registered_target_path$image_basename"
	   echo 'Wrapped_image_path' $Wrapped_image_path  #Wrapped_image_path /home/rtm/scratch/rtm/data/MedImagepreprocess/RegisteredImages/iDB_target_registered2/Wrap_sub-0027_space-pet_FLAIR.nii.gz

      ANTS 3 -m CC[$mni_image,$full_image_name_SS,1,2] -i 10x50x50x20 -o $template_path -t SyN[0.25] -r Gauss[3,0]
      WarpImageMultiTransform 3 $image_name $Wrapped_image_path -R $mni_image $template_wrapped_path $template_Affine_path

fi















