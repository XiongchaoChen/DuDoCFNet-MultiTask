# Dual-Domain Coarse-to-Fine Progressive Network for Simultaneous Denoising, Limited-View Reconstruction, and Attenuation Correction of Cardiac SPECT

Xiongchao Chen, Bo Zhou, Xueqi Guo, Huidong Xie, Qiong Liu, James S. Duncan, Albert J. Sinusas, Chi Liu

### Overview
![image](IMAGE/Figure_1.png)

### Projection-domain framework
![image](IMAGE/Figure_2.png)

### Image-domain framework
![image](IMAGE/Figure_3.png)

This repository contains the PyTorch implementation of the Dual-Domain Coarse-to-Fine Progressive Network (DuDoCFNet).


### Citation 
If you use this code for your research or project, please cite:

    xxx (Under review ...)



 ### Environment and Dependencies
 Requirements:
 * Python 3.6.10
 * Pytorch 1.2.0
 * numpy 1.19.2
 * scipy
 * scikit-image
 * h5py
 * tqdm

Our code has been tested with Python 3.6.10, Pytorch 1.2.0, CUDA: 10.0.130 on Ubuntu 18.04.6.

 ### Dataset Setup
    Data
    ├── train                # contains training files
    |   ├── data1.h5
    |       ├── Amap.mat  
    |       ├── Recon_LD_LA_EM.mat
    |       ├── Proj_FD_FA_EM.mat  
    |       ├── Proj_LD_FA_EM.mat
    |       ├── Proj_LD_LA_EM.mat
    |   └── ...  
    | 
    ├── valid               # contains validation files
    |   ├── data1.h5
    |       ├── Amap.mat  
    |       ├── Recon_LD_LA_EM.mat
    |       ├── Proj_FD_FA_EM.mat  
    |       ├── Proj_LD_FA_EM.mat
    |       ├── Proj_LD_LA_EM.mat
    |   └── ...  
    |
    ├── test                # contains testing files
    |   ├── data1.h5
    |       ├── Amap.mat  
    |       ├── Recon_LD_LA_EM.mat
    |       ├── Proj_FD_FA_EM.mat  
    |       ├── Proj_LD_FA_EM.mat
    |       ├── Proj_LD_LA_EM.mat
    |   └── ...  
    └── ...  

where \
`Amap`: CT-derived attenuation maps with a size of 72 x 72 x 40. \
`Recon_LD_LA_EM`: reconstructed lose-dose and limited-angle images with a size of 72 x 72 x 40. \
`Proj_FD_FA_EM`: full-dose and full-angle projections with a size of 32 x 32 x 20. \
`Proj_LD_FA_EM`: low-dose and full-angle projections with a size of 32 x 32 x 20. \
`Proj_LD_LA_EM`: lose-dose and limited-angle projections with a size of 32 x 32 x 20. \
`Mask_Proj`: binary projection mask with a size of 32 x 32 x 20. 1 refers to the central limited-angle regions. \



