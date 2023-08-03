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


### To Run the Code
Sample training/testing scripts are provided at the root folder as `train.sh` and `test.sh`.

- Train the model 
```bash
python train.py --experiment_name 'train_1' --model_type 'model_cnn' --data_root './xxx' --norm 'BN' --net_filter 32 --n_denselayer 4 --growth_rate 16 --lr_G1 1e-3 --lr_G2 1e-4 --step_size 1 --gamma 0.99 --n_epochs 150 --batch_size 2 --eval_epochs 5 --snapshot_epochs 5 --gpu_ids 0
```
where \
`--experiment_name` experiment name for the code, and save all the training results in this under this "experiment_name" folder. \
`--model_type`: model type used (default convolutional neural networks). \
`--data_root`: the path of the dataset. \
`--norm`: batch normalization in the CNN modules (default: 'BN'). \
`--net_filter`: num of filters in the densely connected layers (default: 32). \
`--n_denselayer`: num of the densely connected layers (default: 4). \
`--growth_rate`: growth rate of the densely connected layers (default: 16). \
`--lr_G1`: learning rate of projection-domain modules (default: 1e-3). \
`--lr_G2`: learning rate of image-domain modules (default: 1e-3). 
`--step_size`: num of epoch for learning rate decay .\
`--gamma`: learning decay rate. \
`--n_epochs`: num of epochs of training. \
`--batch_size`: training batch size. \
`--test_epochs`: number of epochs for periodic validation. \
`--save_epochs`: number of epochs for saving trained model. \
`--gpu_ids`: GPU configuration.


- Test the model 
```bash
python test.py --resume './outputs/train_1/checkpoints/model_xx.pt' --experiment_name 'test_1_xx' --model_type 'model_cnn' --data_root '../xxx' --norm 'BN' --net_filter 32 --n_denselayer 4 --growth_rate 16 --batch_size 2 --gpu_ids 0
```
where \
`--resume`: the path of the model to be tested. \
`--resume_epoch`: training epoch of the model to be tested. 

### Data Availability
The original dataset in this study is available from the corresponding author (chi.liu@yale.edu) upon reasonable request and approval of Yale University. 

### Contact 
If you have any questions, please file an issue or directly contact the author:
```
Xiongchao Chen: xiongchao.chen@yale.edu, cxiongchao9587@gmail.com
```









