import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *
from utils.function import *

# ----------------------- Training Dataset ---------------------
class CardiacSPECT_Train(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_train
        self.n_patch = opts.n_patch_train
        # self.AUG = opts.AUG

        self.data_dir = os.path.join(self.root, 'train')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.Amap_all = []
        self.Recon_LD_LA_EM_all = []
        self.Proj_FD_FA_EM_all = []
        self.Proj_LD_FA_EM_all = []
        self.Proj_LD_LA_EM_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                Amap = f['Amap'][...].transpose(2, 1, 0)
                Recon_LD_LA_EM  = f['Recon_LD_LA_EM'][...].transpose(2, 1, 0)
                Proj_FD_FA_EM  = f['Proj_FD_FA_EM'][...].transpose(2, 1, 0)
                Proj_LD_FA_EM  = f['Proj_LD_FA_EM'][...].transpose(2, 1, 0)
                Proj_LD_LA_EM  = f['Proj_LD_LA_EM'][...].transpose(2, 1, 0)

            # create the random index for cropping patches
            X_template = Amap
            indexes = get_random_patch_indexes(data=X_template, patch_size=[72, 72, 40], num_patches=1, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=Amap, indexes=indexes, patch_size=[72, 72, 40], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Amap_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Recon_LD_LA_EM, indexes=indexes, patch_size=[72, 72, 40], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Recon_LD_LA_EM_all.append(X_patches)

            # create the random index for cropping patches
            X_template = Proj_FD_FA_EM
            indexes = get_random_patch_indexes(data=X_template, patch_size=[32, 32, 20], num_patches=1, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=Proj_FD_FA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_FD_FA_EM_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Proj_LD_FA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_LD_FA_EM_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Proj_LD_LA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_LD_LA_EM_all.append(X_patches)

        self.Amap_all = np.concatenate(self.Amap_all, 0)
        self.Recon_LD_LA_EM_all = np.concatenate(self.Recon_LD_LA_EM_all, 0)
        self.Proj_FD_FA_EM_all = np.concatenate(self.Proj_FD_FA_EM_all, 0)
        self.Proj_LD_FA_EM_all = np.concatenate(self.Proj_LD_FA_EM_all, 0)
        self.Proj_LD_LA_EM_all = np.concatenate(self.Proj_LD_LA_EM_all, 0)

    def __getitem__(self, index):
        Amap = self.Amap_all[index, ...]
        Recon_LD_LA_EM = self.Recon_LD_LA_EM_all[index, ...]
        Proj_FD_FA_EM  = self.Proj_FD_FA_EM_all[index, ...]
        Proj_LD_FA_EM  = self.Proj_LD_FA_EM_all[index, ...]
        Proj_LD_LA_EM  = self.Proj_LD_LA_EM_all[index, ...]

        Amap = torch.from_numpy(Amap.copy())
        Recon_LD_LA_EM = torch.from_numpy(Recon_LD_LA_EM.copy())
        Proj_FD_FA_EM = torch.from_numpy(Proj_FD_FA_EM.copy())
        Proj_LD_FA_EM = torch.from_numpy(Proj_LD_FA_EM.copy())
        Proj_LD_LA_EM = torch.from_numpy(Proj_LD_LA_EM.copy())

        return {'Amap': Amap,
                'Recon_LD_LA_EM': Recon_LD_LA_EM,
                'Proj_FD_FA_EM': Proj_FD_FA_EM,
                'Proj_LD_FA_EM': Proj_LD_FA_EM,
                'Proj_LD_LA_EM': Proj_LD_LA_EM,
                'opts_drop': True}

    def __len__(self):
        return self.Amap_all.shape[0]




# ----------------------- Testing Dataset ---------------------
class CardiacSPECT_Test(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_test
        self.n_patch = opts.n_patch_test
        # self.test_pad = opts.test_pad

        self.data_dir = os.path.join(self.root, 'test')  # Attention difference here
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.Amap_all = []
        self.Recon_LD_LA_EM_all = []
        self.Proj_FD_FA_EM_all = []
        self.Proj_LD_FA_EM_all = []
        self.Proj_LD_LA_EM_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                Amap = f['Amap'][...].transpose(2, 1, 0)
                Recon_LD_LA_EM = f['Recon_LD_LA_EM'][...].transpose(2, 1, 0)
                Proj_FD_FA_EM = f['Proj_FD_FA_EM'][...].transpose(2, 1, 0)
                Proj_LD_FA_EM = f['Proj_LD_FA_EM'][...].transpose(2, 1, 0)
                Proj_LD_LA_EM = f['Proj_LD_LA_EM'][...].transpose(2, 1, 0)

            # create the random index for cropping patches
            X_template = Amap
            indexes = get_random_patch_indexes(data=X_template, patch_size=[72, 72, 40], num_patches=1, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=Amap, indexes=indexes, patch_size=[72, 72, 40], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Amap_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Recon_LD_LA_EM, indexes=indexes, patch_size=[72, 72, 40], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Recon_LD_LA_EM_all.append(X_patches)

            # create the random index for cropping patches
            X_template = Proj_FD_FA_EM
            indexes = get_random_patch_indexes(data=X_template, patch_size=[32, 32, 20], num_patches=1, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=Proj_FD_FA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_FD_FA_EM_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Proj_LD_FA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_LD_FA_EM_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Proj_LD_LA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_LD_LA_EM_all.append(X_patches)

        self.Amap_all = np.concatenate(self.Amap_all, 0)
        self.Recon_LD_LA_EM_all = np.concatenate(self.Recon_LD_LA_EM_all, 0)
        self.Proj_FD_FA_EM_all = np.concatenate(self.Proj_FD_FA_EM_all, 0)
        self.Proj_LD_FA_EM_all = np.concatenate(self.Proj_LD_FA_EM_all, 0)
        self.Proj_LD_LA_EM_all = np.concatenate(self.Proj_LD_LA_EM_all, 0)

    def __getitem__(self, index):
        Amap = self.Amap_all[index, ...]
        Recon_LD_LA_EM = self.Recon_LD_LA_EM_all[index, ...]
        Proj_FD_FA_EM = self.Proj_FD_FA_EM_all[index, ...]
        Proj_LD_FA_EM = self.Proj_LD_FA_EM_all[index, ...]
        Proj_LD_LA_EM = self.Proj_LD_LA_EM_all[index, ...]

        Amap = torch.from_numpy(Amap.copy())
        Recon_LD_LA_EM = torch.from_numpy(Recon_LD_LA_EM.copy())
        Proj_FD_FA_EM = torch.from_numpy(Proj_FD_FA_EM.copy())
        Proj_LD_FA_EM = torch.from_numpy(Proj_LD_FA_EM.copy())
        Proj_LD_LA_EM = torch.from_numpy(Proj_LD_LA_EM.copy())

        return {'Amap': Amap,
                'Recon_LD_LA_EM': Recon_LD_LA_EM,
                'Proj_FD_FA_EM': Proj_FD_FA_EM,
                'Proj_LD_FA_EM': Proj_LD_FA_EM,
                'Proj_LD_LA_EM': Proj_LD_LA_EM,
                'opts_drop': False}

    def __len__(self):
        return self.Amap_all.shape[0]





# ----------------------- Validation Dataset ---------------------
class CardiacSPECT_Valid(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_valid
        self.n_patch = opts.n_patch_valid
        # self.valid_pad = opts.valid_pad

        self.data_dir = os.path.join(self.root, 'valid')  # Attention difference here
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.Amap_all = []
        self.Recon_LD_LA_EM_all = []
        self.Proj_FD_FA_EM_all = []
        self.Proj_LD_FA_EM_all = []
        self.Proj_LD_LA_EM_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                Amap = f['Amap'][...].transpose(2, 1, 0)
                Recon_LD_LA_EM = f['Recon_LD_LA_EM'][...].transpose(2, 1, 0)
                Proj_FD_FA_EM = f['Proj_FD_FA_EM'][...].transpose(2, 1, 0)
                Proj_LD_FA_EM = f['Proj_LD_FA_EM'][...].transpose(2, 1, 0)
                Proj_LD_LA_EM = f['Proj_LD_LA_EM'][...].transpose(2, 1, 0)
                Mask_Proj = f['Mask_Proj'][...].transpose(2, 1, 0)

            # create the random index for cropping patches
            X_template = Amap
            indexes = get_random_patch_indexes(data=X_template, patch_size=[72, 72, 40], num_patches=1, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=Amap, indexes=indexes, patch_size=[72, 72, 40], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Amap_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Recon_LD_LA_EM, indexes=indexes, patch_size=[72, 72, 40], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Recon_LD_LA_EM_all.append(X_patches)

            # create the random index for cropping patches
            X_template = Proj_FD_FA_EM
            indexes = get_random_patch_indexes(data=X_template, patch_size=[32, 32, 20], num_patches=1, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=Proj_FD_FA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_FD_FA_EM_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Proj_LD_FA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_LD_FA_EM_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=Proj_LD_LA_EM, indexes=indexes, patch_size=[32, 32, 20], padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.Proj_LD_LA_EM_all.append(X_patches)

        self.Amap_all = np.concatenate(self.Amap_all, 0)
        self.Recon_LD_LA_EM_all = np.concatenate(self.Recon_LD_LA_EM_all, 0)
        self.Proj_FD_FA_EM_all = np.concatenate(self.Proj_FD_FA_EM_all, 0)
        self.Proj_LD_FA_EM_all = np.concatenate(self.Proj_LD_FA_EM_all, 0)
        self.Proj_LD_LA_EM_all = np.concatenate(self.Proj_LD_LA_EM_all, 0)

    def __getitem__(self, index):
        Amap = self.Amap_all[index, ...]
        Recon_LD_LA_EM = self.Recon_LD_LA_EM_all[index, ...]
        Proj_FD_FA_EM = self.Proj_FD_FA_EM_all[index, ...]
        Proj_LD_FA_EM = self.Proj_LD_FA_EM_all[index, ...]
        Proj_LD_LA_EM = self.Proj_LD_LA_EM_all[index, ...]

        Amap = torch.from_numpy(Amap.copy())
        Recon_LD_LA_EM = torch.from_numpy(Recon_LD_LA_EM.copy())
        Proj_FD_FA_EM = torch.from_numpy(Proj_FD_FA_EM.copy())
        Proj_LD_FA_EM = torch.from_numpy(Proj_LD_FA_EM.copy())
        Proj_LD_LA_EM = torch.from_numpy(Proj_LD_LA_EM.copy())

        return {'Amap': Amap,
                'Recon_LD_LA_EM': Recon_LD_LA_EM,
                'Proj_FD_FA_EM': Proj_FD_FA_EM,
                'Proj_LD_FA_EM': Proj_LD_FA_EM,
                'Proj_LD_LA_EM': Proj_LD_LA_EM,
                'opts_drop': False}

    def __len__(self):
        return self.Amap_all.shape[0]


if __name__ == '__main__':
    pass
