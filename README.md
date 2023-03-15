# MaskMixAdv: Scribble-supervised Medical Image Segmentation via MaskMix-based Siamese Network and Shape-aware Adversarial Learning
1) MaskMixAdv for the first time designs a Mask-based Mixup strategy (MaskMix) for scribble-supervised medical image segmentation. MaskMix introduces image-level and feature-level perturbations to one sample for data augmentation, replacing the graphical-based or the traditional cross-sample mixup approaches.

2) MaskMixAdv for the first time designs a dual-branches siamese network, trained by segmentation and reconstruction in different branches based on the MaskMix strategy. Pseudo labels are generated by integrating the two-branch prediction results through complementary binary masks, which can further boost 3D Dice.

3) MaskMixAdv learns to regularize the generated pseudo labels via shape-aware adversarial learning to incorporate additional shape priors, which can reduce the Hausdorff distance.

<p align="center"><img width="=100%" src="imgs/framework.png" /></p>

## Packages Requirements
- Hardware: PC with NVIDIA 1080T GPU. (others are alternative.)
- Software: *Ubuntu 18.04*, *CUDA 10.0.130*, *pytorch 1.3.0*, *Python 3.6.9* (others are alternative.)
- Some important required packages include:
  - `torchvision`
  - `tensorboardX`
  - `scikit-learn`
  - `glob`
  - `matplotlib`
  - `skimage`
  - `medpy`
  - `tqdm`
  - `nibabel`
  - `Efficientnet-Pytorch`: `pip install efficientnet_pytorch`
  - Other basic python packages such as `Numpy`, `Scikit-image`, `SimpleITK`, `Scipy`, `cv2` ......

# Dataset
Datasets and more details can be found from the following links. 
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
* The Scribble annotations of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data).
* The data processing code in [Here](../code/dataloaders/acdc_data_processing.py), the pre-processed ACDC data in [Here](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).
* The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html).
* Please organize the dataset as the following structure:
```
ACDC/
  -- ACDC_training_slices/
      --patient001_frame01_slice_0.h5
      ...
  -- ACDC_training_volumes/
      --patient001_frame01.h5
      ...
MSCMR_dataset/
  -- train/
      --images/
      --labels/
        --patient001_frame01.h5
  ...
```

# Usage

1. Clone this project.
```
git clone ***************
cd MaskMixAdv/code
```
2. Data pre-processing os used or the processed data.
```
python dataloaders/acdc_data_processing.py
```
3. Train the model
```
./train_wss.sh
```

4. Test the model
```
python test_2D_fully.py --sup_type <scribble or label> --exp ACDC/<the_trained_model_path> --model <model_name>
```

# Implemented methods
* Some of the results shown are referenced from those reported in the [CVPR 2022 & Supplementary](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_CycleMix_A_Holistic_Strategy_for_Medical_Image_Segmentation_From_Scribble_CVPR_2022_paper.html) and [Weakly-supervised benchmark](https://link.springer.com/chapter/10.807/978-3-031-16431-6_50).
* [**pCE**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tang_Normalized_Cut_Loss_CVPR_2018_paper.pdf) : [train_weakly_supervised_pCE_2D.py](./code/train_weakly_supervised_pCE_2D.py)
* [**pCE + TV**](https://arxiv.org/pdf/1605.01368.pdf) : [train_weakly_supervised_pCE_TV_2D.py](./code/train_weakly_supervised_pCE_TV_2D.py)
* [**pCE + Entropy Minimization**](https://arxiv.org/pdf/2111.02403.pdf) : [train_weakly_supervised_pCE_Entropy_Mini_2D.py](./code/train_weakly_supervised_pCE_Entropy_Mini_2D.py)
* [**pCE + GatedCRFLoss**](https://github.com/LEONOB2014/GatedCRFLoss) : [train_weakly_supervised_pCE_GatedCRFLoss_2D.py](./code/train_weakly_supervised_pCE_GatedCRFLoss_2D.py)
* [**pCE + Random Walker**](http://vision.cse.psu.edu/people/chenpingY/paper/grady2006random.pdf) : [train_weakly_supervised_pCE_random_walker_2D.py](./code/train_weakly_supervised_pCE_random_walker_2D.py)
* [**pCE + MumfordShah_Loss**](https://arxiv.org/pdf/1804.02872.pdf) : [train_weakly_supervised_pCE_MumfordShah_Loss_2D.py](./code/train_weakly_supervised_pCE_MumfordShah_Loss_2D.py)
* [**Scribble2Label**](https://arxiv.org/pdf/2006.12880.pdf)
* [**USTM**](https://www.sciencedirect.com/science/article/pii/S003132032805215) : [train_weakly_supervised_ustm_2D.py](./code/train_weakly_supervised_ustm_2D.py)
* [**WSL4MIS**](https://github.com/Luoxd1996/WSL4MIS) : [train_weakly_supervised_pCE_WSL4MIS.py](./code/train_weakly_supervised_pCE_WSL4MIS.py)
* [**MaskMixAdv**](ours) : [train_weakly_supervised_pCE_MaskMixAdv.py](./code/train_weakly_supervised_pCE_MaskMixAdv.py)

## Major results from our work
1. **MaskMixAdv achieved the best performance among all weakly-supervised learning SOTA methods on the MRI cardiac segmentation.**

<p align="center"><img width="90%" src="imgs/compare_result.png" /></p>


2. **The discrepancy between MaskMixAdv trained on scribbles and the supervised method trained on dense annotations was minor. Compared with previous methods that generated misshapen predictions, MaskMixAdv generated more realistic and reasonable segmentation masks.**

<p align="center"><img width="80%" src="imgs/results1.png" /></p>

3. **Ablation study indicated that:**
**1) Scribble-supervised methods performed poorly when only L_pCE and L_DpCE were applied. In contrast, performance improved after introducing L_rec.**
**2) The 3D Dice of L_DpCE + L_rec was still not satisfied. The pseudo labeling component (L_pse) improved it.**
**3) The HD_95 of L_DpCE + L_rec + L_pse was still far below that of supervised methods. Adversarial learning (L_adv) reduced the HD_95 discrepancy between the weakly and fully supervised methods.**

<p align="center"><img width="90%" src="imgs/ablation_result.png" /></p>

4. **MaskMixAdv achieved the best performance among all image-level and feature-level perturbations on the MRI cardiac segmentation.**

<p align="center"><img width="80%" src="imgs/mixup.png" /></p>

5. **Visual demonstration of the results of the proposed MaskMixAdv and other methods for cardiac MR data perturbation. Only perturbations at the image level are shown here, and perturbations at the feature level are similar and thus omitted. For the Mixup-based approach, we introduce white outlines to easily distinguish the multi-sample mixing process. Note the scribbles shown here are bolded for ease of viewing.**

<p align="center"><img width="80%" src="imgs/perturbation.png" /></p>

6. **MaskMixAdv was extended to point-supervised cardiac segmentation on the ACDC dataset. The minor performance gap demonstrated the scalability of our work. We will release the code for generating point annotations before publication in this repo.**

<p align="center"><img width="100%" src="imgs/points.png" /></p>

## Acknowledgement
Anonymous
## Reference
Anonymous
## License
Anonymous

#### ** We hope that in the light of our study, the medical imaging and computer vision community will benefit from the use of more powerful weakly-supervised models. **
