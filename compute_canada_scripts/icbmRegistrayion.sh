#!/bin/bash




new_target_images_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/target.txt
new_atlas_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas.txt
new_atlas_label_folder=/home/rtm/scratch/rtm/data/labelFusion/paths/atlas_label.txt

registered_target_path=/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/iDB_target_registered/
registered_atlas_path=/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered/
registered_atlas_label_path=/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_label_registered/

mni_image=/home/rtm/scratch/rtm/data/icbm/mni_icbm152_t1_tal_nlin_sym_09c.nii.gz
mni_GT=/home/rtm/scratch/rtm/data/icbm/mni_icbm152_t1_lateralventricles.nii.gz


mkdir -p $registered_atlas_path $registered_target_path registered_atlas_label_path
echo 'paths file created'


while read image_name<&3 && read label_name<&4
    do

      input_basename="$(basename -- $image_name)"
      echo 'input_basename' $input_basename #OAS1_0061_MR1.nii
      input_basename_without_postfix=$(echo $input_basename | cut -d . -f 1 -)
      echo 'input_basename_without_postfix' $input_basename_without_postfix  #OAS1_0061_MR1


      img_postfix='template_'
      template_path="$registered_atlas_path$img_postfix$input_basename_without_postfix"
      echo 'template_path' $template_path # /home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered/template_OAS1_0061_MR1

      postfixWrap='Wrap.nii.gz'
      template_wrapped_path="$template_path$postfixWrap"
      echo 'template_wrapped_path' $template_wrapped_path #/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered/template_OAS1_0061_MR1Wrap.nii.gz

      postfixAffine='Affine.txt'
      template_Affine_path="$template_path$postfixAffine"
      echo 'template_Affine_path' $template_Affine_path #/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered/template_OAS1_0061_MR1Affine.txt

      img_postfix='Wrap_'
	   image_basename="$img_postfix$input_basename"
	   Wrapped_image_path="$registered_atlas_path$image_basename"
	   echo 'Wrapped_image_path' $Wrapped_image_path  #//home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered/Wrap_OAS1_0061_MR1.nii


	   label_basename="$(basename -- $label_name)"
	   img_postfix='Wrap_'
	   label_wrap_basename="$img_postfix$label_basename"
	   Wrapped_label_path="$registered_atlas_label_path$label_wrap_basename"
	   echo 'Wrapped_label_path' $Wrapped_label_path #/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_label_registered/Wrap_OAS1_0061_MR1_seg.nii

      neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular \
      ANTS 3 -m CC[$mni_image,$image_name,1,2] -i 10x50x50x20 -o $template_path -t SyN[0.25] -r Gauss[3,0] \
      &&  WarpImageMultiTransform 3 $image_name $Wrapped_image_path -R $mni_image $template_wrapped_path $template_Affine_path \
      &&  WarpImageMultiTransform 3 $label_name $Wrapped_label_path -R $mni_image $template_wrapped_path $template_Affine_path


done 3<$new_atlas_folder  4<$new_atlas_label_folder




while read image_name<&3
    do

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
	   Wrapped_image_path="$registered_atlas_path$image_basename"
	   echo 'Wrapped_image_path' $Wrapped_image_path  #/home/rtm/scratch/rtm/data/labelFusion/RegisteredImages/OASIS_atlas_registered/Wrap_sub-0001_space-pet_FLAIR.nii.gz


#
      neurogliaSubmit -I  /project/6055004/tools/singularity/khanlab-neuroglia-dwi-master-v1.4.1.simg   \
      -j Regular \
      ANTS 3 -m CC[$mni_image,$image_name,1,2] -i 10x50x50x20 -o $template_path -t SyN[0.25] -r Gauss[3,0] \
      &&  WarpImageMultiTransform 3 $image_name $Wrapped_image_path -R $mni_image $template_wrapped_path $template_Affine_path \

done 3<$new_target_images_folder






















