import os
from abc import ABC

import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from scipy.special import entr
from utils.function import *
import pdb

from networks import *
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, mse, nmse, nmae
from skimage.metrics import structural_similarity as ssim
from utils.data_patch_util import *
import scipy.io as scio
from scipy.sparse import coo_matrix


class CNNModel(nn.Module):
    def __init__(self, opts):
        super(CNNModel, self).__init__()

        self.loss_names = []  # list
        self.networks = []  # list
        self.optimizers = []  # list
        # self.lr_G0 = opts.lr_G0
        self.lr_G1 = opts.lr_G1  # Proj
        self.lr_G2 = opts.lr_G2  # Amap
        self.loss_weight = opts.loss_weight
        # self.net_filter = opts.net_filter

        # Loss Name
        # self.loss_names += ['loss_G0']
        self.loss_names += ['loss_G1f_1']  # mediate loss
        self.loss_names += ['loss_G1_1']
        self.loss_names += ['loss_G2g_1']
        self.loss_names += ['loss_G2b_1']
        self.loss_names += ['loss_G2_1']

        self.loss_names += ['loss_G1f_2']
        self.loss_names += ['loss_G1_2']
        self.loss_names += ['loss_G2g_2']
        self.loss_names += ['loss_G2b_2']
        self.loss_names += ['loss_G2_2']

        self.loss_names += ['loss_G1f_3']
        self.loss_names += ['loss_G1_3']
        self.loss_names += ['loss_G2g_3']
        self.loss_names += ['loss_G2b_3']
        self.loss_names += ['loss_G2_3']

        self.loss_names += ['loss_G1f_4']
        self.loss_names += ['loss_G1_4']
        self.loss_names += ['loss_G2g_4']
        self.loss_names += ['loss_G2b_4']
        self.loss_names += ['loss_G2_4']

        # Network
        # self.net_G0   = get_generator('DuRDN3',    opts)  # Amap-Net (Prior)
        self.net_G1_1 = get_generator('TSFuseSE_TL0', opts, 1)
        self.net_G2_1 = get_generator('TSFuseSE', opts, 1)
        self.net_G1_2 = get_generator('TSFuseSE_TL', opts, 2)
        self.net_G2_2 = get_generator('TSFuseSE', opts, 2)
        self.net_G1_3 = get_generator('TSFuseSE_TL', opts, 3)
        self.net_G2_3 = get_generator('TSFuseSE', opts, 3)
        self.net_G1_4 = get_generator('TSFuseSE_TL', opts, 4)
        self.net_G2_4 = get_generator('TSFuseSE', opts, 4)
        # self.networks.append(self.net_G0)
        self.networks.append(self.net_G1_1)
        self.networks.append(self.net_G2_1)
        self.networks.append(self.net_G1_2)
        self.networks.append(self.net_G2_2)
        self.networks.append(self.net_G1_3)
        self.networks.append(self.net_G2_3)
        self.networks.append(self.net_G1_4)
        self.networks.append(self.net_G2_4)

        # Optimizer
        # self.optimizer_G0   = torch.optim.Adam(self.net_G0.parameters(),   lr=self.lr_G0, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G1_1 = torch.optim.Adam(self.net_G1_1.parameters(), lr=self.lr_G1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G2_1 = torch.optim.Adam(self.net_G2_1.parameters(), lr=self.lr_G2, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G1_2 = torch.optim.Adam(self.net_G1_2.parameters(), lr=self.lr_G1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G2_2 = torch.optim.Adam(self.net_G2_2.parameters(), lr=self.lr_G2, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G1_3 = torch.optim.Adam(self.net_G1_3.parameters(), lr=self.lr_G1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G2_3 = torch.optim.Adam(self.net_G2_3.parameters(), lr=self.lr_G2, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G1_4 = torch.optim.Adam(self.net_G1_4.parameters(), lr=self.lr_G1, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_G2_4 = torch.optim.Adam(self.net_G2_4.parameters(), lr=self.lr_G2, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        # self.optimizers.append(self.optimizer_G0)
        self.optimizers.append(self.optimizer_G1_1)
        self.optimizers.append(self.optimizer_G2_1)
        self.optimizers.append(self.optimizer_G1_2)
        self.optimizers.append(self.optimizer_G2_2)
        self.optimizers.append(self.optimizer_G1_3)
        self.optimizers.append(self.optimizer_G2_3)
        self.optimizers.append(self.optimizer_G1_4)
        self.optimizers.append(self.optimizer_G2_4)

        # Loss Function
        self.criterion = nn.L1Loss()  # L1 loss function.py

        # Options
        self.opts = opts


    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))  # Choose GPU for CUDA computing; For input setting

    def system_matrix(self):
        # Read in the sparse matrix in coo_matrix format
        SM = coo_matrix(scio.loadmat('./sm/sm601_20.mat')['sm'])

        # Extract the values, indices, and shape
        values = torch.FloatTensor(SM.data)
        indices = torch.LongTensor(np.vstack((SM.row, SM.col)))
        shape = torch.Size(SM.shape)

        # Build the system matrix in the torch sparse format
        self.SM = torch.sparse.FloatTensor(indices, values, shape).to(self.device).float().unsqueeze(0).unsqueeze(0)  #  [1, 1, 32*32*20, 72*72*40], nnz = 28768054
        # self.SM = torch.sparse.FloatTensor(indices, values, shape).to(self.device).float().unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  #  [B, 1, 32*32*20, 72*72*40]

        # convert the sparse SM to dense SM
        self.SM_dense = self.SM.to_dense()  # [1, 1, 32*32*20, 72*72*40]


    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]


    # LR decay can be realized here
    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]


    def set_input(self, data):
        self.Amap = data['Amap'].to(self.device).float()
        self.Recon_LD_LA  = data['Recon_LD_LA_EM'].to(self.device).float()

        self.Proj_FD_FA  = data['Proj_FD_FA_EM'].to(self.device).float()
        self.Proj_LD_FA  = data['Proj_LD_FA_EM'].to(self.device).float()
        self.Proj_LD_LA  = data['Proj_LD_LA_EM'].to(self.device).float()

        self.opts_drop = data['opts_drop'][0].numpy()  # Training: True; Testing: False
        self.sm_size = self.Amap.size(0)  # Batch size


    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, name))  # get self.loss_G_L1
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        # Set inp as require_grad_
        self.Proj_LD_LA.requires_grad_(True)
        self.Recon_LD_LA.requires_grad_(True)

        # Iter1
        self.Proj_LD_FA_pred1, self.Proj_FD_FA_pred1 = self.net_G1_1(self.Proj_LD_LA)
        self.Proj_FD_FA_pred1_norm = self.Proj_FD_FA_pred1 / self.Proj_FD_FA_pred1.detach().mean() * self.Proj_LD_LA.detach().mean()  # norm
        self.Proj_FD_FA_pred1_BP = torch.matmul(self.Proj_FD_FA_pred1.permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 1, 32 * 32 * 20), self.SM_dense).reshape(self.sm_size, 1, 40, 72, 72).permute(0, 1, 4, 3, 2).rot90(2, [2, 3]).flip(4)  # [B,1, 72, 72, 40]
        self.Proj_FD_FA_pred1_BP_norm = self.Proj_FD_FA_pred1_BP / self.Proj_FD_FA_pred1_BP.detach().mean()

        self.Amap_pred1_g, self.Amap_pred1_b, self.Amap_pred1 = self.net_G2_1(self.Recon_LD_LA, self.Proj_FD_FA_pred1_BP_norm)
        self.Amap_pred1_norm = self.Amap_pred1 / self.Amap_pred1.detach().mean()
        self.Amap_pred1_FP = torch.matmul(self.SM_dense, self.Amap_pred1.flip(4).rot90(-2, [2, 3]).permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 40 * 72 * 72, 1)).reshape(self.sm_size, 1, 20, 32, 32).permute(0, 1, 4, 3, 2)  # [B,1, 32, 32, 20]
        self.Amap_pred1_FP_norm = self.Amap_pred1_FP / self.Amap_pred1_FP.detach().mean() *  self.Proj_LD_LA.detach().mean()  # norm

        # Iter 2
        self.Proj_LD_FA_pred2, self.Proj_FD_FA_pred2 = self.net_G1_2(torch.cat((self.Proj_LD_LA, self.Proj_FD_FA_pred1_norm), 1), self.Amap_pred1_FP_norm)
        self.Proj_FD_FA_pred2_norm = self.Proj_FD_FA_pred2 / self.Proj_FD_FA_pred2.detach().mean() * self.Proj_LD_LA.detach().mean()  # norm
        self.Proj_FD_FA_pred2_BP = torch.matmul(self.Proj_FD_FA_pred2.permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 1, 32 * 32 * 20), self.SM_dense).reshape(self.sm_size, 1, 40, 72, 72).permute(0, 1, 4, 3, 2).rot90(2, [2, 3]).flip(4)  # [B,1, 72, 72, 40]
        self.Proj_FD_FA_pred2_BP_norm = self.Proj_FD_FA_pred2_BP / self.Proj_FD_FA_pred2_BP.detach().mean()

        self.Amap_pred2_g, self.Amap_pred2_b, self.Amap_pred2 = self.net_G2_2(torch.cat((self.Recon_LD_LA, self.Amap_pred1_norm), 1), self.Proj_FD_FA_pred2_BP_norm)
        self.Amap_pred2_norm = self.Amap_pred2 / self.Amap_pred2.detach().mean()
        self.Amap_pred2_FP = torch.matmul(self.SM_dense, self.Amap_pred2.flip(4).rot90(-2, [2, 3]).permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 40 * 72 * 72, 1)).reshape(self.sm_size, 1, 20, 32, 32).permute(0, 1, 4, 3, 2)  # [B,1, 32, 32, 20]
        self.Amap_pred2_FP_norm = self.Amap_pred2_FP / self.Amap_pred2_FP.detach().mean() * self.Proj_LD_LA.detach().mean()  # norm

        # Iter 3
        self.Proj_LD_FA_pred3, self.Proj_FD_FA_pred3 = self.net_G1_3(torch.cat((self.Proj_LD_LA, self.Proj_FD_FA_pred1_norm, self.Proj_FD_FA_pred2_norm), 1), self.Amap_pred2_FP_norm)
        self.Proj_FD_FA_pred3_norm = self.Proj_FD_FA_pred3 / self.Proj_FD_FA_pred3.detach().mean() * self.Proj_LD_LA.detach().mean()  # norm
        self.Proj_FD_FA_pred3_BP = torch.matmul(self.Proj_FD_FA_pred3.permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 1, 32 * 32 * 20), self.SM_dense).reshape(self.sm_size, 1, 40, 72, 72).permute(0, 1, 4, 3, 2).rot90(2, [2, 3]).flip(4)  # [B,1, 72, 72, 40]
        self.Proj_FD_FA_pred3_BP_norm = self.Proj_FD_FA_pred3_BP / self.Proj_FD_FA_pred3_BP.detach().mean()

        self.Amap_pred3_g, self.Amap_pred3_b, self.Amap_pred3 = self.net_G2_3(torch.cat((self.Recon_LD_LA, self.Amap_pred1_norm, self.Amap_pred2_norm), 1), self.Proj_FD_FA_pred3_BP_norm)
        self.Amap_pred3_norm = self.Amap_pred3 / self.Amap_pred3.detach().mean()
        self.Amap_pred3_FP = torch.matmul(self.SM_dense, self.Amap_pred3.flip(4).rot90(-2, [2, 3]).permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 40 * 72 * 72, 1)).reshape(self.sm_size, 1, 20, 32, 32).permute(0, 1, 4, 3, 2)  # [B,1, 32, 32, 20]
        self.Amap_pred3_FP_norm = self.Amap_pred3_FP / self.Amap_pred3_FP.detach().mean() * self.Proj_LD_LA.detach().mean()  # norm

        # Iter 4
        self.Proj_LD_FA_pred4, self.Proj_FD_FA_pred4 = self.net_G1_4(torch.cat((self.Proj_LD_LA, self.Proj_FD_FA_pred1_norm, self.Proj_FD_FA_pred2_norm, self.Proj_FD_FA_pred3_norm), 1), self.Amap_pred3_FP_norm)
        self.Proj_FD_FA_pred4_norm = self.Proj_FD_FA_pred4 / self.Proj_FD_FA_pred4.detach().mean() * self.Proj_LD_LA.detach().mean()  # norm
        self.Proj_FD_FA_pred4_BP = torch.matmul(self.Proj_FD_FA_pred4.permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 1, 32 * 32 * 20), self.SM_dense).reshape(self.sm_size, 1, 40, 72, 72).permute(0, 1, 4, 3, 2).rot90(2, [2, 3]).flip(4)  # [B,1, 72, 72, 40]
        self.Proj_FD_FA_pred4_BP_norm = self.Proj_FD_FA_pred4_BP / self.Proj_FD_FA_pred4_BP.detach().mean()

        self.Amap_pred4_g, self.Amap_pred4_b, self.Amap_pred4 = self.net_G2_4(torch.cat((self.Recon_LD_LA, self.Amap_pred1_norm, self.Amap_pred2_norm, self.Amap_pred3_norm), 1), self.Proj_FD_FA_pred4_BP_norm)
        self.Amap_pred4_FP = torch.matmul(self.SM_dense, self.Amap_pred4.flip(4).rot90(-2, [2, 3]).permute(0, 1, 4, 3, 2).reshape(self.sm_size, 1, 40 * 72 * 72, 1)).reshape(self.sm_size, 1, 20, 32, 32).permute(0, 1, 4, 3, 2)  # [B,1, 32, 32, 20]



    def update(self):
        # Zero Gradient
        # self.optimizer_G0.zero_grad()
        self.optimizer_G1_1.zero_grad()
        self.optimizer_G2_1.zero_grad()
        self.optimizer_G1_2.zero_grad()
        self.optimizer_G2_2.zero_grad()
        self.optimizer_G1_3.zero_grad()
        self.optimizer_G2_3.zero_grad()
        self.optimizer_G1_4.zero_grad()
        self.optimizer_G2_4.zero_grad()

        # Calculate Loss
        # loss_G0   = self.criterion(self.Amap_pred0, self.Amap)
        loss_G1f_1  = self.criterion(self.Proj_LD_FA_pred1, self.Proj_LD_FA)
        loss_G1_1   = self.criterion(self.Proj_FD_FA_pred1, self.Proj_FD_FA)
        loss_G2g_1  = self.criterion(self.Amap_pred1_g, self.Amap)
        loss_G2b_1  = self.criterion(self.Amap_pred1_b, gradient_img(self.Amap))
        loss_G2_1   = self.criterion(self.Amap_pred1, self.Amap)

        loss_G1f_2  = self.criterion(self.Proj_LD_FA_pred2, self.Proj_LD_FA)
        loss_G1_2   = self.criterion(self.Proj_FD_FA_pred2, self.Proj_FD_FA)
        loss_G2g_2  = self.criterion(self.Amap_pred2_g, self.Amap)
        loss_G2b_2  = self.criterion(self.Amap_pred2_b, gradient_img(self.Amap))
        loss_G2_2   = self.criterion(self.Amap_pred2, self.Amap)

        loss_G1f_3  = self.criterion(self.Proj_LD_FA_pred3, self.Proj_LD_FA)
        loss_G1_3   = self.criterion(self.Proj_FD_FA_pred3, self.Proj_FD_FA)
        loss_G2g_3  = self.criterion(self.Amap_pred3_g, self.Amap)
        loss_G2b_3  = self.criterion(self.Amap_pred3_b, gradient_img(self.Amap))
        loss_G2_3   = self.criterion(self.Amap_pred3, self.Amap)

        loss_G1f_4  = self.criterion(self.Proj_LD_FA_pred4, self.Proj_LD_FA)
        loss_G1_4   = self.criterion(self.Proj_FD_FA_pred4, self.Proj_FD_FA)
        loss_G2g_4  = self.criterion(self.Amap_pred4_g, self.Amap)
        loss_G2b_4  = self.criterion(self.Amap_pred4_b, gradient_img(self.Amap))
        loss_G2_4   = self.criterion(self.Amap_pred4, self.Amap)

        # self.loss_G0   = loss_G0.item()
        self.loss_G1f_1 = loss_G1f_1.item()
        self.loss_G1_1  = loss_G1_1.item()
        self.loss_G2g_1 = loss_G2g_1.item()
        self.loss_G2b_1 = loss_G2b_1.item()
        self.loss_G2_1  = loss_G2_1.item()

        self.loss_G1f_2 = loss_G1f_2.item()
        self.loss_G1_2  = loss_G1_2.item()
        self.loss_G2g_2 = loss_G2g_2.item()
        self.loss_G2b_2 = loss_G2b_2.item()
        self.loss_G2_2  = loss_G2_2.item()

        self.loss_G1f_3 = loss_G1f_3.item()
        self.loss_G1_3  = loss_G1_3.item()
        self.loss_G2g_3 = loss_G2g_3.item()
        self.loss_G2b_3 = loss_G2b_3.item()
        self.loss_G2_3  = loss_G2_3.item()

        self.loss_G1f_4 = loss_G1f_4.item()
        self.loss_G1_4  = loss_G1_4.item()
        self.loss_G2g_4 = loss_G2g_4.item()
        self.loss_G2b_4 = loss_G2b_4.item()
        self.loss_G2_4  = loss_G2_4.item()

        # Backward and update
        total_loss = ( (loss_G1f_1 + loss_G1f_2 + loss_G1f_3 + loss_G1f_4) + (loss_G1_1 + loss_G1_2 + loss_G1_3 + loss_G1_4) ) + \
                     ( (loss_G2g_1 + loss_G2g_2 + loss_G2g_3 + loss_G2g_4) + (loss_G2b_1 + loss_G2b_2 + loss_G2b_3 + loss_G2b_4) + (loss_G2_1 + loss_G2_2 + loss_G2_3 + loss_G2_4) ) * self.loss_weight

        total_loss.backward()

        # self.optimizer_G0.step()
        self.optimizer_G1_1.step()
        self.optimizer_G2_1.step()
        self.optimizer_G1_2.step()
        self.optimizer_G2_2.step()
        self.optimizer_G1_3.step()
        self.optimizer_G2_3.step()
        self.optimizer_G1_4.step()
        self.optimizer_G2_4.step()


    @property  # Only for this function.py
    def loss_summary(self):
        message = ''
        message += 'loss1:{:.4e}, loss2:{:.4e}'.format(self.loss_G1_4, self.loss_G2_4)
        return message

    # learning rate decay
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()  # learning rate update
        self.lr_G1 = self.optimizers[0].param_groups[0]['lr']
        self.lr_G2 = self.optimizers[1].param_groups[0]['lr']


    def save(self, filename, epoch, total_iter):  # Save the net/optimizer state data
        state = {}  # dict
        # state['net_G0']   = self.net_G0.module.state_dict()
        state['net_G1_1'] = self.net_G1_1.module.state_dict()
        state['net_G2_1'] = self.net_G2_1.module.state_dict()
        state['net_G1_2'] = self.net_G1_2.module.state_dict()
        state['net_G2_2'] = self.net_G2_2.module.state_dict()
        state['net_G1_3'] = self.net_G1_3.module.state_dict()
        state['net_G2_3'] = self.net_G2_3.module.state_dict()
        state['net_G1_4'] = self.net_G1_4.module.state_dict()
        state['net_G2_4'] = self.net_G2_4.module.state_dict()

        # state['opt_G0']   = self.optimizer_G0.state_dict()
        state['opt_G1_1'] = self.optimizer_G1_1.state_dict()
        state['opt_G2_1'] = self.optimizer_G2_1.state_dict()
        state['opt_G1_2'] = self.optimizer_G1_2.state_dict()
        state['opt_G2_2'] = self.optimizer_G2_2.state_dict()
        state['opt_G1_3'] = self.optimizer_G1_3.state_dict()
        state['opt_G2_3'] = self.optimizer_G2_3.state_dict()
        state['opt_G1_4'] = self.optimizer_G1_4.state_dict()
        state['opt_G2_4'] = self.optimizer_G2_4.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))


    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        # self.net_G0.module.load_state_dict(checkpoint['net_G0'])
        self.net_G1_1.module.load_state_dict(checkpoint['net_G1_1'])
        self.net_G2_1.module.load_state_dict(checkpoint['net_G2_1'])
        self.net_G1_2.module.load_state_dict(checkpoint['net_G1_2'])
        self.net_G2_2.module.load_state_dict(checkpoint['net_G2_2'])
        self.net_G1_3.module.load_state_dict(checkpoint['net_G1_3'])
        self.net_G2_3.module.load_state_dict(checkpoint['net_G2_3'])
        self.net_G1_4.module.load_state_dict(checkpoint['net_G1_4'])
        self.net_G2_4.module.load_state_dict(checkpoint['net_G2_4'])

        if train:
            # self.optimizer_G0.load_state_dict(checkpoint['opt_G0'])
            self.optimizer_G1_1.load_state_dict(checkpoint['opt_G1_1'])
            self.optimizer_G2_1.load_state_dict(checkpoint['opt_G2_1'])
            self.optimizer_G1_2.load_state_dict(checkpoint['opt_G1_2'])
            self.optimizer_G2_2.load_state_dict(checkpoint['opt_G2_2'])
            self.optimizer_G1_3.load_state_dict(checkpoint['opt_G1_3'])
            self.optimizer_G2_3.load_state_dict(checkpoint['opt_G2_3'])
            self.optimizer_G1_4.load_state_dict(checkpoint['opt_G1_4'])
            self.optimizer_G2_4.load_state_dict(checkpoint['opt_G2_4'])

        print('Loaded {}'.format(checkpoint_file))
        return checkpoint['epoch'], checkpoint['total_iter']


    # -------------- Evaluation, Calculate PSNR ---------------
    def evaluate(self, loader):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Evaluating images ...')

        # For calculating metrics
        avg_nmse_1 = AverageMeter()
        avg_nmae_1 = AverageMeter()
        avg_psnr_1 = AverageMeter()
        avg_ssim_1 = AverageMeter() # Proj

        avg_nmse_2 = AverageMeter()
        avg_nmae_2 = AverageMeter()
        avg_psnr_2 = AverageMeter()
        avg_ssim_2 = AverageMeter() # Amap

        for data in val_bar:
            self.set_input(data)  # [batch_szie=1, 1, 48, 72, 72]
            self.forward()

            # Non-megativity & Mean-normalization
            self.Proj_FD_FA_pred4[self.Proj_FD_FA_pred4 < 0] = 0  # non-negativity
            self.Amap_pred4[self.Amap_pred4 < 0] = 0

            # Calculate the metrics; z_range can be used to calculate the mean
            # Proj
            nmse_1 = nmse(self.Proj_FD_FA_pred4, self.Proj_FD_FA)
            nmae_1 = nmae(self.Proj_FD_FA_pred4, self.Proj_FD_FA)
            psnr_1 = psnr(self.Proj_FD_FA_pred4, self.Proj_FD_FA)
            ssim_1 = ssim(self.Proj_FD_FA_pred4[0, 0, ...].cpu().numpy(), self.Proj_FD_FA[0, 0, ...].cpu().numpy())
            avg_nmse_1.update(nmse_1)
            avg_nmae_1.update(nmae_1)
            avg_psnr_1.update(psnr_1)
            avg_ssim_1.update(ssim_1)

            # Amap
            nmse_2 = nmse(self.Amap_pred4, self.Amap)
            nmae_2 = nmae(self.Amap_pred4, self.Amap)
            psnr_2 = psnr(self.Amap_pred4, self.Amap)
            ssim_2 = ssim(self.Amap_pred4[0, 0, ...].cpu().numpy(), self.Amap[0, 0, ...].cpu().numpy())
            avg_nmse_2.update(nmse_2)
            avg_nmae_2.update(nmae_2)
            avg_psnr_2.update(psnr_2)
            avg_ssim_2.update(ssim_2)

            # Descrip show NMSE, NMAE, SSIM here
            message = 'NMSE_Proj: {:4f}, NMSE_Amap: {:4f}, NMAE_Proj: {:4f}, NMAE_Amap: {:4f}'.format(avg_nmse_1.avg, avg_nmse_2.avg,   avg_nmae_1.avg, avg_nmae_2.avg)
            val_bar.set_description(desc=message)

        # Calculate the average metrics
        self.nmse_1 = avg_nmse_1.avg
        self.nmae_1 = avg_nmae_1.avg
        self.psnr_1 = avg_psnr_1.avg
        self.ssim_1 = avg_ssim_1.avg

        self.nmse_2 = avg_nmse_2.avg
        self.nmae_2 = avg_nmae_2.avg
        self.psnr_2 = avg_psnr_2.avg
        self.ssim_2 = avg_ssim_2.avg


    # --------------- Save the images ------------------------------
    def save_images(self, loader, folder):
        val_bar = tqdm(loader)
        val_bar.set_description(desc='Saving images ...')

        # Load data for each batch
        index = 0
        for data in val_bar:
            index += 1
            self.set_input(data)  # [batch_szie=1, 1, 64, 64, 64]
            self.forward()

            # Non-megativity & Mean-normalization
            self.Proj_FD_FA_pred4[self.Proj_FD_FA_pred4 < 0] = 0  # non-negativity
            self.Amap_pred4[self.Amap_pred4 < 0] = 0

            # --------------- Mkdir folder -------------------
            # Original 64x64x64 images
            if not os.path.exists(os.path.join(folder, 'Proj_FD_FA_pred')):
                os.mkdir(os.path.join(folder, 'Proj_FD_FA_pred'))
            if not os.path.exists(os.path.join(folder, 'Proj_FD_FA')):
                os.mkdir(os.path.join(folder, 'Proj_FD_FA'))
            if not os.path.exists(os.path.join(folder, 'Proj_LD_FA_pred')):
                os.mkdir(os.path.join(folder, 'Proj_LD_FA_pred'))
            if not os.path.exists(os.path.join(folder, 'Proj_LD_FA')):
                os.mkdir(os.path.join(folder, 'Proj_LD_FA'))
            if not os.path.exists(os.path.join(folder, 'Proj_LD_LA')):
                os.mkdir(os.path.join(folder, 'Proj_LD_LA'))

            if not os.path.exists(os.path.join(folder, 'Amap_pred')):
                os.mkdir(os.path.join(folder, 'Amap_pred'))
            if not os.path.exists(os.path.join(folder, 'Amap')):
                os.mkdir(os.path.join(folder, 'Amap'))
            if not os.path.exists(os.path.join(folder, 'Recon_LD_LA')):
                os.mkdir(os.path.join(folder, 'Recon_LD_LA'))

            # save image
            save_nii(self.Proj_FD_FA_pred4.squeeze().cpu().numpy(), os.path.join(folder, 'Proj_FD_FA_pred', 'Proj_FD_FA_pred_' + str(index) + '.nii'))
            save_nii(self.Proj_FD_FA.squeeze().cpu().numpy(), os.path.join(folder, 'Proj_FD_FA', 'Proj_FD_FA_' + str(index) + '.nii'))
            save_nii(self.Proj_LD_FA_pred4.squeeze().cpu().numpy(), os.path.join(folder, 'Proj_LD_FA_pred', 'Proj_LD_FA_pred_' + str(index) + '.nii'))
            save_nii(self.Proj_LD_FA.squeeze().cpu().numpy(), os.path.join(folder, 'Proj_LD_FA', 'Proj_LD_FA_' + str(index) + '.nii'))
            save_nii(self.Proj_LD_LA.squeeze().cpu().numpy(), os.path.join(folder, 'Proj_LD_LA', 'Proj_LD_LA_' + str(index) + '.nii'))

            save_nii(self.Amap_pred4.squeeze().cpu().numpy(),   os.path.join(folder, 'Amap_pred',  'Amap_pred_'   + str(index) + '.nii'))
            save_nii(self.Amap.squeeze().cpu().numpy(),   os.path.join(folder, 'Amap',  'Amap_'   + str(index) + '.nii'))
            save_nii(self.Recon_LD_LA.squeeze().cpu().numpy(),   os.path.join(folder, 'Recon_LD_LA',  'Recon_LD_LA_'   + str(index) + '.nii'))












