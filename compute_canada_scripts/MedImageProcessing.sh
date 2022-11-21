#!/bin/bash


N4_atlas_path=/home/rtm/scratch/rtm/data/labelFusion/N4BiasFieldCorrected/atlas_OASIS2/
N4_target_path=/home/rtm/scratch/rtm/data/labelFusion/N4BiasFieldCorrected/target_IBD2/

SS_target_path=/home/rtm/scratch/rtm/data/labelFusion/SkullStrippedimages/iDB_SS2/
SS_atlas_path=/home/rtm/scratch/rtm/data/labelFusion/SkullStrippedimages/atlasOASIS2/


registered_target_path=/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/iDB_target_registered2/
registered_atlas_path=/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered2/
registered_atlas_label_path=/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_label_registered2/

mni_image=/home/rtm/scratch/rtm/data/icbm/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz
mni_GT=/home/rtm/scratch/rtm/data/icbm/mni_icbm152_t1_lateralventricles.nii.gz


mkdir -p $registered_atlas_path $registered_target_path $registered_atlas_label_path $SS_target_path $SS_atlas_path $N4_target_path $N4_atlas_path
echo 'paths file created'
level_flag=0 # atlas images process
if [[ $# -eq 1 ]];
  then
    level_flag=0 # target images process
fi



# reading command line arguments
while getopts "c:d:f:g:h:j:k:l:m:o:p:q:r:t:u:v:w:x:y:z:" OPT
  do
  case $OPT in
      a) #atlas
    image_name=$OPTARG
   echo "$USAGE"
   exit 0
   ;;
      l) #label image
   label_name=$OPTARG
   if [[ $DOQSUB -gt 5 ]];
     then
       echo " DOQSUB must be an integer value (0=serial, 1=SGE qsub, 2=try pexec, 3=XGrid, 4=PBS qsub, 5=SLURM ) you passed  -c $DOQSUB "
       exit 1
     fi
   ;;
      t) #targetimage
   image_name=$OPTARG
   if [[ ${DIM} -ne 2 && $DIM -ne 3 ]];
     then
       echo " Dimensionality is only valid for 2 or 3.  You passed -d $DIM."
       exit 1
     fi
   ;;
   esac
done



if [[ $level_flag == 0 ]]; then

      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #OAS1_0061_MR1.nii

      img_postfix='N4correct_'
	   image_name_N4="$img_postfix$input_basename"
	   full_image_name_N4="$N4_atlas_path$image_name_N4"
	   echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4


      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] \
      -i $image_name -o $full_image_name_N4




      img_postfix='SkullStripped_'
	   image_basename="$img_postfix$input_basename"
	   full_image_name_SS="$SS_atlas_path$image_basename"

      bet $full_image_name_N4 $full_image_name_SS -m -f 0.15 -g 0.1 -R -B



      img_postfix='Registered_'
	   image_basename="$img_postfix$input_basename"
	   full_image_name_registered="$registered_atlas_path$image_basename"

      flirt -in $full_image_name_SS -ref $mni_image -out $full_image_name_registered

    ###############################################3

fi




if [[ $evel_flag == 1 ]]; then


      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #OAS1_0061_MR1.nii

      img_postfix='N4correct_'
	   image_name_N4="$img_postfix$input_basename"
	   full_image_name_N4="$N4_target_path$image_name_N4"
	   echo '$image_name ' $image_name ' $full_image_name_N4 ' $full_image_name_N4


      N4BiasFieldCorrection -d 3 -r 1 -b [200,2] -c [400x200x100x40,0.0] \
      -i $image_name -o $full_image_name_N4




      img_postfix='SkullStripped_'
	   image_basename="$img_postfix$input_basename"
	   full_image_name_SS="$SS_target_path$image_basename"

      bet $full_image_name_N4 $full_image_name_SS -m -f 0.15 -g 0.1 -R -B


      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #sub-0001_space-pet_FLAIR.nii.gz
      input_basename_without_postfix=$(echo $input_basename | cut -d . -f 1 -)
      echo 'input_basename_without_postfix' $input_basename_without_postfix  #sub-0001_space-pet_FLAIR


      img_postfix='template_'
      template_path="$registered_target_path$img_postfix$input_basename_without_postfix"
      echo 'template_path' $template_path # /home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/iDB_target_registered/template_sub-0001_space-pet_FLAIR

      postfixWrap='Wrap.nii.gz'
      template_wrapped_path="$template_path$postfixWrap"
      echo 'template_wrapped_path' $template_wrapped_path #home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/iDB_target_registered/template_sub-0001_space-pet_FLAIRWrap.nii.gz

      postfixAffine='Affine.txt'
      template_Affine_path="$template_path$postfixAffine"
      echo 'template_Affine_path' $template_Affine_path #/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/iDB_target_registered/template_sub-0001_space-pet_FLAIRAffine.txt

      img_postfix='Wrap_'
	   image_basename="$img_postfix$input_basename"
	   Wrapped_image_path="$registered_target_path$image_basename"
	   echo 'Wrapped_image_path' $Wrapped_image_path  #/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered/Wrap_sub-0001_space-pet_FLAIR.nii.gz


      ANTS 3 -m CC[$mni_image,$full_image_name_SS,1,2] -i 10x50x50x20 -o $template_path -t SyN[0.25] -r Gauss[3,0]
      WarpImageMultiTransform 3 $image_name $Wrapped_image_path -R $mni_image $template_wrapped_path $template_Affine_path

fi






















