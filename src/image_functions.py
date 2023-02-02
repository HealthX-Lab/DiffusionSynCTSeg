
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from src.transformer_model import MyTransformer
from monai.utils import first
def save_image(opt,val_files):
    transformer_obj_val = MyTransformer(opt, 'val')
    val_data_transformes = transformer_obj_val.get_data_transformer()
    check_ds = Dataset(data=val_files, transform=val_data_transformes)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    MRI, CT, label = (check_data["MRI"][0][0], check_data["CT"][0][0], check_data["label"][0][0])
    print(f"MRI shape: {MRI.shape}, CT shape: {CT.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title('MRI')
    axs[0].imshow(MRI[:, :, 50], cmap="gray")
    axs[1].set_title('CT')
    axs[1].imshow(CT[:, :, 50], cmap="gray")
    axs[2].set_title('label')
    axs[2].imshow(label[:, :, 50] )
    os.makedirs(opt.val_template_save_path.split('/')[1],exist_ok = True)
    # Save the figure
    plt.savefig(opt.val_template_save_path)
    plt.close()

