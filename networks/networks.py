import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, get_nonlinearity
from networks.SE import *
from utils.data_patch_util import *
import pdb


class TSFuseSE_TL0(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=3, growth_rate=16, norm='None'):
        super(TSFuseSE_TL0, self).__init__()
        # -------- First Layer ------------
        self.Conv_0 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        self.SE_1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_3   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.Up_up2   = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.SE_up2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.Up_up1   = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SE_up1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        # self.Conv_outf = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)
        self.SAM = SAM(n_filters)


        # -------- Second Layer ----------
        self.Conv_T0 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.Conv_SAM = nn.Conv3d(2*n_filters, n_filters, kernel_size=3, padding=1, bias=True)
        self.SE_T0   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_T3   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_T2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_T1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.Conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)


    def forward(self, inp_):    # Dual-Channel Input
        # -------- First layer --------
        Conv_0 = self.Conv_0(inp_)
        SE_1 = self.SE_1(Conv_0)

        Down_2 = F.avg_pool3d(SE_1, 2)
        SE_2 = self.SE_2(Down_2)

        Down_3 = F.avg_pool3d(SE_2, 2)
        SE_3 = self.SE_3(Down_3)

        Up_up2 = self.Up_up2(SE_3)
        SE_up2 = self.SE_up2(Up_up2 + SE_2)

        Up_up1 = self.Up_up1(SE_up2)
        SE_up1 = self.SE_up1(Up_up1 + SE_1)

        # Conv_outf = self.Conv_outf(SE_up1)
        SAM_att, Conv_outf = self.SAM(SE_up1)


        # --------- Second Layer ------------
        Conv_T0 = self.Conv_T0(Conv_outf)
        Conv_SAM = self.Conv_SAM(torch.cat((Conv_T0, SAM_att), 1))
        SE_T0 = self.SE_T0(Conv_SAM)

        SE_T3 = self.SE_T3(SE_T0 + F.interpolate(SE_3, scale_factor=4))

        SE_T2 = self.SE_T2(SE_T3 + F.interpolate(SE_2, scale_factor=2) + F.interpolate(SE_up2, scale_factor=2))

        SE_T1 = self.SE_T1(SE_T2 + SE_1 + SE_up1)

        Conv_out = self.Conv_out(SE_T1) + Conv_outf

        return Conv_outf, Conv_out





class TSFuseSE_TL(nn.Module):
    def __init__(self, n_channels = 1, n_filters=32, n_denselayer=3, growth_rate=16, norm='None'):
        super(TSFuseSE_TL, self).__init__()
        # ------ First Layer ---------
        self.Conv_0 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        self.ConvTS_1 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.cSFE_1 = cSFE(n_filters)
        self.SE_1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.ConvTS_2 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.cSFE_2 = cSFE(n_filters)
        self.SE_2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.ConvTS_3 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.cSFE_3 = cSFE(n_filters)
        self.SE_3   = SEConv(n_filters, n_denselayer, growth_rate, norm)


        self.Up_up2   = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.SE_up2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.Up_up1   = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SE_up1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        # self.Conv_outf = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)
        self.SAM = SAM(n_filters)


        # -------- Second Layer ----------
        self.Conv_T0 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.Conv_SAM = nn.Conv3d(2*n_filters, n_filters, kernel_size=3, padding=1, bias=True)
        self.SE_T0   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_T3   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_T2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.SE_T1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.Conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)


    def forward(self, inp_, inp_TS):    # Dual-Channel Input
        # ----- First Layer ----------
        Conv_0 = self.Conv_0(inp_)
        ConvTS_1 = self.ConvTS_1(inp_TS)
        SE_1 = self.SE_1(self.cSFE_1(Conv_0, ConvTS_1))

        Down_2 = F.avg_pool3d(SE_1, 2)
        ConvTS_2 = self.ConvTS_2(F.avg_pool3d(inp_TS, 2))
        SE_2 = self.SE_2(self.cSFE_2(Down_2, ConvTS_2))

        Down_3 = F.avg_pool3d(SE_2, 2)
        ConvTS_3 = self.ConvTS_3(F.avg_pool3d(inp_TS, 4))
        SE_3 = self.SE_3(self.cSFE_3(Down_3, ConvTS_3))

        Up_up2 = self.Up_up2(SE_3)
        SE_up2 = self.SE_up2(Up_up2 + SE_2)

        Up_up1 = self.Up_up1(SE_up2)
        SE_up1 = self.SE_up1(Up_up1 + SE_1)

        # Conv_outf = self.Conv_outf(SE_up1)
        SAM_att, Conv_outf = self.SAM(SE_up1)


        # --------- Second Layer ------------
        Conv_T0 = self.Conv_T0(Conv_outf)
        Conv_SAM = self.Conv_SAM(torch.cat((Conv_T0, SAM_att), 1))
        SE_T0 = self.SE_T0(Conv_SAM)

        SE_T3 = self.SE_T3(SE_T0 + F.interpolate(SE_3, scale_factor=4))

        SE_T2 = self.SE_T2(SE_T3 + F.interpolate(SE_2, scale_factor=2) + F.interpolate(SE_up2, scale_factor=2))

        SE_T1 = self.SE_T1(SE_T2 + SE_1 + SE_up1)

        Conv_out = self.Conv_out(SE_T1) + Conv_outf

        return Conv_outf, Conv_out




class TSFuseSE(nn.Module):
    def __init__(self, n_channels = 1, n_filters=32, n_denselayer=3, growth_rate=16, norm='None'):
        super(TSFuseSE, self).__init__()
        # -- Downsampling --
        self.Conv_0 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        self.ConvTS_1 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.cSFE_1 = cSFE(n_filters)
        self.SE_1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.ConvTS_2 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.cSFE_2 = cSFE(n_filters)
        self.SE_2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.ConvTS_3 = nn.Conv3d(1, n_filters, kernel_size=3, padding=1, bias=True)
        self.cSFE_3 = cSFE(n_filters)
        self.SE_3   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        # -- Upsampling: General Structure --
        self.Up_up2   = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.SE_up2   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.Up_up1   = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SE_up1   = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.Conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)


        # -- Upsampling: Boundary --
        self.UpX_up2 = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SEX_up2 = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.UpX_up1 = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SEX_up1 = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.ConvX_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)


        # -- Final Estimation --
        self.sSFE = sSFE(n_filters, n_denselayer, growth_rate, norm)
        self.ConvY_0 = nn.Conv3d(2, n_filters, kernel_size=3, padding=1, bias=True)
        self.ConvY_1 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1, bias=True)
        # self.SEY_1 = SEConv(n_filters, n_denselayer, growth_rate, norm)
        self.ConvY_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)




    def forward(self, inp_, inp_TS):    # Dual-Channel Input
        # -- Downsample --
        Conv_0 = self.Conv_0(inp_)
        ConvTS_1 = self.ConvTS_1(inp_TS)
        SE_1 = self.SE_1(self.cSFE_1(Conv_0, ConvTS_1))

        Down_2 = F.avg_pool3d(SE_1, 2)
        ConvTS_2 = self.ConvTS_2(F.avg_pool3d(inp_TS, 2))
        SE_2 = self.SE_2(self.cSFE_2(Down_2, ConvTS_2))

        Down_3 = F.avg_pool3d(SE_2, 2)
        ConvTS_3 = self.ConvTS_3(F.avg_pool3d(inp_TS, 4))
        SE_3 = self.SE_3(self.cSFE_3(Down_3, ConvTS_3))


        # -- Upsampling: General Structure --
        Up_up2 = self.Up_up2(SE_3)
        SE_up2 = self.SE_up2(Up_up2 + SE_2)

        Up_up1 = self.Up_up1(SE_up2)
        SE_up1 = self.SE_up1(Up_up1 + SE_1)

        Conv_out_general = self.Conv_out(SE_up1)


        # -- Upsampling: Boundary Estimation ---
        UpX_up2 = self.UpX_up2(SE_3)
        SEX_up2 = self.SEX_up2(UpX_up2 + SE_2)

        UpX_up1 = self.UpX_up1(SEX_up2)
        SEX_up1 = self.SEX_up1(UpX_up1 + SE_1)

        Conv_out_boundary = self.ConvX_out(SEX_up1)


        # -- Final Estimation --
        ConvY_sSFE = self.sSFE(Conv_out_general, Conv_out_boundary)
        ConvY_0 = self.ConvY_0(ConvY_sSFE)
        ConvY_1 = self.ConvY_1(ConvY_0)
        Conv_out_final = self.ConvY_out(ConvY_1)

        return Conv_out_general, Conv_out_boundary, Conv_out_final



# Channel SE: for fusing the multi-channel inputs
class cSFE(nn.Module):
    def __init__(self, n_channels_extract = 32):
        super(cSFE, self).__init__()
        self.avg_pool_ch1 = nn.AdaptiveAvgPool3d((1,1,1))
        self.avg_pool_ch2 = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc_comb = nn.Linear(2*n_channels_extract, n_channels_extract, bias=True)
        self.fc_ch1 = nn.Linear(n_channels_extract, n_channels_extract, bias=True)
        self.fc_ch2 = nn.Linear(n_channels_extract, n_channels_extract, bias=True)

    def forward(self, inp_ch1, inp_ch2):
        # Basic Information
        batch_size, n_channels, D, H, W = inp_ch1.size()

        # (1) Spatial-Squeeze + Channel-Excitation
        squeeze_ch1 = self.avg_pool_ch1(inp_ch1).view(batch_size, n_channels)  # [B, C, 1,1,1] to [B, C]
        squeeze_ch2 = self.avg_pool_ch2(inp_ch2).view(batch_size, n_channels)  # [B, C, 1,1,1] to [B, C]
        squeeze_comb = torch.cat((squeeze_ch1, squeeze_ch2), 1)  # [B, C*2]

        # Fully connected layers
        fc_comb = self.fc_comb(squeeze_comb)
        fc_ch1 = torch.sigmoid(self.fc_ch1(fc_comb))
        fc_ch2 = torch.sigmoid(self.fc_ch2(fc_comb))

        # Multiplication
        inp_ch1_scSE = torch.mul(inp_ch1, fc_ch1.view(batch_size, n_channels, 1, 1, 1))
        inp_ch2_scSE = torch.mul(inp_ch2, fc_ch2.view(batch_size, n_channels, 1, 1, 1))  # [B, C, D,H,W]

        inp_cat = inp_ch1_scSE + inp_ch2_scSE

        return inp_cat


# Spatial SE: for fusing the general and boundary information
class sSFE(nn.Module):
    def __init__(self, n_filters=32, n_denselayer=3, growth_rate=16, norm='None'):
        super(sSFE, self).__init__()
        self.conv_volume_ch1 = nn.Conv3d(1, n_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_volume_ch2 = nn.Conv3d(1, n_filters, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv_comb = nn.Conv3d(n_filters*2, n_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.SE_comb = SEConv(n_filters, n_denselayer, growth_rate, norm)

        self.conv_adjust_ch1 = nn.Conv3d(n_filters, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_adjust_ch2 = nn.Conv3d(n_filters, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_adjust_ch1 = nn.BatchNorm3d(1)
        self.bn_adjust_ch2 = nn.BatchNorm3d(1)

    def forward(self, inp_ch1, inp_ch2):
        # Basic Information
        batch_size, n_channels, D, H, W = inp_ch1.size()

        # Feature Expansion
        volume_ch1 = self.conv_volume_ch1(inp_ch1)
        volume_ch2 = self.conv_volume_ch2(inp_ch2)
        volume_comb = torch.cat((volume_ch1, volume_ch2), 1)

        # Fusion Layer
        conv_comb = self.conv_comb(volume_comb)
        SE_comb = self.SE_comb(conv_comb)

        conv_adjust_ch1 = 2*torch.sigmoid(self.bn_adjust_ch1(self.conv_adjust_ch1(SE_comb)))
        conv_adjust_ch2 = 2*torch.sigmoid(self.bn_adjust_ch2(self.conv_adjust_ch2(SE_comb)))

        # Multiplication
        inp_ch1_csSE = torch.mul(inp_ch1, conv_adjust_ch1.view(batch_size, 1, D, H, W)) + inp_ch1  # Changed here
        inp_ch2_csSE = torch.mul(inp_ch2, conv_adjust_ch2.view(batch_size, 1, D, H, W)) + inp_ch2  # Changed here

        out_cSFE = torch.cat((inp_ch1_csSE, inp_ch2_csSE), 1)

        return out_cSFE




# SAM (Supervised Attention Module)
class SAM(nn.Module):
    def __init__(self, n_filters = 32):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv3d(1, n_filters, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv3d(n_filters, n_filters, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_pred):
        out_pred = self.conv1(inp_pred)

        attention_weights = self.sigmoid(self.conv2(out_pred))
        out_SAM = self.conv3(inp_pred) * attention_weights + inp_pred

        return out_SAM, out_pred



# SE Convolution Module
class SEConv(nn.Module):
    def __init__(self, n_filters=32, n_denselayer=3, growth_rate=16, norm='None'):
        super(SEConv, self).__init__()
        self.RDB = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE = ChannelSpatialSELayer3D(n_filters*1, norm='None')

    def forward(self, inp_):
        out_ = self.SE(self.RDB(inp_))
        out_SE = out_
        return out_SE



# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x # Residual
        return out



# Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)

    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out






'''
spatial-channel Squeeze and Excite Residual Dense UNet (depth = 4)
'''
class scSERDUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None'):
        super(scSERDUNet, self).__init__()
        # Channel-wise weight self-recalibration
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(n_channels, n_channels*2, bias=True)
        self.fc2 = nn.Linear(n_channels*2, n_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Input layers
        # self.dropout = dropout
        self.conv_in = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB4 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE4 = ChannelSpatialSELayer3D(n_filters * 1, norm='None')

        # decode
        self.up3 = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.fuse_up3 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up3 = RDB(n_filters * 1, n_denselayer, growth_rate, norm)
        self.SE_up3 = ChannelSpatialSELayer3D(n_filters * 1, norm='None')

        self.up2 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up2 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up1 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # Channel-wise Self-recalibration
        batch_size, num_channels, D, H, W = x.size()
        squeeze_tensor = self.avg_pool(x)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        # encode
        down1 = self.conv_in(output_tensor)
        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)

        down2 = F.avg_pool3d(SE1, 2)
        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)

        down3 = F.avg_pool3d(SE2, 2)
        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)

        down4 = F.avg_pool3d(SE3, 2)
        RDB4 = self.RDB4(down4)
        SE4 = self.SE4(RDB4)

        # # Dropout, function.py at testing phase; avoid overfitting
        # if self.dropout & opts_drop:
        #     SE4 = F.dropout(SE4, p=0.3)

        # decode
        up3 = self.up3(SE4)
        RDB_up3 = self.RDB_up3(self.fuse_up3(torch.cat((up3, SE3), 1)))
        SE_up3 = self.SE_up3(RDB_up3)

        up2 = self.up2(SE_up3)
        RDB_up2 = self.RDB_up2(self.fuse_up2(torch.cat((up2, SE2), 1)))
        SE_up2 = self.SE_up2(RDB_up2)

        up1 = self.up1(SE_up2)
        RDB_up1 = self.RDB_up1(self.fuse_up1(torch.cat((up1, SE1), 1)))
        SE_up1 = self.SE_up1(RDB_up1)

        output = self.conv_out(SE_up1)
        return output




'''
spatial-channel Squeeze and Excite Residual Dense UNet (depth = 3)
'''
class scSERDUNet3(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None'):
        super(scSERDUNet3, self).__init__()
        # Channel-wise weight self-recalibration
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(n_channels, n_channels * 2, bias=True)
        self.fc2 = nn.Linear(n_channels * 2, n_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Input layer
        # self.dropout = dropout
        self.conv_in = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        # decode
        self.up2 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up2 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up1 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # Channel-wise Self-recalibration
        batch_size, num_channels, D, H, W = x.size()
        squeeze_tensor = self.avg_pool(x)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        # encode
        down1 = self.conv_in(output_tensor)
        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)

        down2 = F.avg_pool3d(SE1, 2)
        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)

        down3 = F.avg_pool3d(SE2, 2)
        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)

        # # Dropout, function.py at testing phase; avoid overfitting
        # if self.dropout & opts_drop:
        #     SE3 = F.dropout(SE3, p=0.3)

        # decode
        up2 = self.up2(SE3)
        RDB_up2 = self.RDB_up2(self.fuse_up2(torch.cat((up2, SE2), 1)))
        SE_up2 = self.SE_up2(RDB_up2)

        up1 = self.up1(SE_up2)
        RDB_up1 = self.RDB_up1(self.fuse_up1(torch.cat((up1, SE1), 1)))
        SE_up1 = self.SE_up1(RDB_up1)

        output = self.conv_out(SE_up1)

        return output


# Weight Initialization for Neural Network Parameters
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)




if __name__ == '__main__':
    pass

