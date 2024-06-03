# Deep learning-based brain ventricle segmentation in Computed Tomography using domain adaptation
# Project Overview
Efficient and accurate brain ventricle segmentation from clinical CT scans is critical for emergency surgeries like ventriculostomy. Our project introduces a novel uncertainty-aware ventricle segmentation technique that leverages diffusion-model-based domain adaptation without the need for CT segmentation ground truths. This method employs the diffusion Schrödinger Bridge and an attention recurrent residual U-Net to utilize unpaired CT and MRI scans for automatic CT segmentation, demonstrating significant improvement over traditional methods. For detailed insights, you can read our paper: [CT-based brain ventricle segmentation via
diffusion Schr¨odinger Bridge without target domain ground truths](https://arxiv.org/pdf/2405.18267)
# Challenges
CT scans lack publicly available, well-annotated databases, unlike MRI, which are essential for developing robust brain segmentation algorithms. Additionally, intuitive confidence measures are needed for segmentation results from automated algorithms like deep learning.

# Solution: Uncertainty-aware Domain Adaptation
We propose an end-to-end uncertainty-aware domain adaptation technique that combines translation models and anatomical segmentation, using unpaired MRI and CT scans without segmentation ground truths.

## Techniques and Models Used:
# Translation Models:
Cycle-Consistent Adversarial Networks (CycleGAN): [CycleGAN GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master)
Contrastive Learning for Unpaired Image-to-Image Translation (CUT): [CUT GitHub](https://github.com/taesungp/contrastive-unpaired-translation)
Unpaired Neural Schrödinger Bridge (UNSB): [UNSB GitHub](https://github.com/cyclomon/UNSB/tree/main)
# Segmentation Models: 
Our segmentation phase employed an attention-based residual recurrent U-Net architecture, compared with traditional U-Net and ResNet. For more information on the base segmentation model, see [R2AU-Net](https://www.hindawi.com/journals/scn/2021/6625688/).
Stability and Consistency: Given CycleGAN's known challenges with stability and structural consistency, we assessed various methods to enhance translation and segmentation performance during our end-to-end training process.
Confidence Measures: We incorporated Monte Carlo dropouts in both MRI-to-CT translation and CT segmentation phases to provide an intuitive interpretation of the segmentation results.
# Inspiration
Our model's design is inspired by [SynSeg-Net](https://github.com/MASILab/SynSeg-Net/tree/df3c26146a36d8ff329917e2d12a79b42bbb6614), aimed at synthesizing and segmenting anatomical structures in medical imaging. Details on this inspiration can be found here: SynSeg-Net GitHub.

# Conclusion
Our innovative approach aims to segment CT ventricles without ground truths, using only MRI and MRI labels. All labels for the test dataset and part of the training were created with the label fusion technique. This strategy enhances the reliability and applicability of CT-based segmentation in clinical settings.
