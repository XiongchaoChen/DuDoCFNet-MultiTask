import torch
import torch.nn.functional as F
import pdb
from utils.data_patch_util import *

# Image gradient
def gradient(img):  # [B,C, X, Y, Z]

    x_step = F.pad(img, [0, 0,  0, 0,  0, 1])[:,:, 1:,  :,  :]
    y_step = F.pad(img, [0, 0,  0, 1,  0, 0])[:,:,  :, 1:,  :]
    z_step = F.pad(img, [0, 1,  0, 0,  0, 0])[:,:,  :,  :, 1:]

    dx, dy, dz= x_step - img, y_step - img, z_step - img
    dx[:, :, -1, :, :] = 0
    dy[:, :, :, -1, :] = 0
    dz[:, :, :, :, -1] = 0

    return dx, dy, dz

# Image gradient loss
def gradient_loss(gen_frames, gt_frames):
    # gradient
    gen_dx, gen_dy, gen_dz = gradient(gen_frames)
    gt_dx, gt_dy, gt_dz = gradient(gt_frames)

    grad_diff_x = torch.abs(gen_dx - gt_dx)
    grad_diff_y = torch.abs(gen_dy - gt_dy)
    grad_diff_z = torch.abs(gen_dz - gt_dz)

    loss_ = torch.mean(grad_diff_x.pow(2) + grad_diff_y.pow(2) + grad_diff_z.pow(2))

    return loss_


# Gradient Image
def gradient_img(img):

    img_dx, img_dy, img_dz = gradient(img)
    grad_img = (torch.abs(img_dx) + torch.abs(img_dy) + torch.abs(img_dz))/3

    return grad_img






