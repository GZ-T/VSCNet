import torch 
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn
from network.modules import Modules as modules



def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)




def pixelshuffle(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])


class VSCNet(nn.Module):
   
    def __init__(self, num_feat=48, num_blocks=30):
        super().__init__()

        self.pixel_unshuffle_layer = nn.PixelUnshuffle(2)
        self.pixel_shuffle_layer = nn.PixelShuffle(2)
        self.upchannel = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body_split = make_layer(modules.VAB, 3, num_feat, 48)
        self.sigmod = nn.Sigmoid()

        self.color_decouple = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=4)
        self.color_global = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.CD_body_merge = make_layer(modules.VAB, 3, num_feat, 48)
        self.CG_body_merge = make_layer(modules.VAB, 3, num_feat, 48)




        self.body_finetune_1 = make_layer(modules.VAB, 3, 48, 48)
        self.finetune_merge_1 = nn.Conv2d(96, 48, 3, 1, 1)
        self.body_finetune_2 = make_layer(modules.VAB, 6, 48, 48)
        self.finetune_merge_2 = nn.Conv2d(96, 48, 3, 1, 1)
        self.LReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        
        self.body_ISP = make_layer(modules.VAB, 12, 48, 48)
        self.conv_ISP_2 = nn.Conv2d(48, 48, 3, 1, 1, groups=1)

        self.upsampler = pixelshuffle(48, 3, upscale_factor=4)



    def forward(self, isp_img, finetune_num=2):


        raw_fea = self.upchannel(isp_img) 
        split_vector = self.body_split(raw_fea) 


        CD_split_cof = self.color_decouple(split_vector)
        CD_split_cof = self.CD_body_merge(CD_split_cof) 
        CG_split_cof = self.color_global(split_vector)
        CG_split_cof = self.CG_body_merge(CG_split_cof) 
        
        
        split_vector = self.sigmod(CD_split_cof+CG_split_cof) 
        split_conf = rearrange(split_vector, 'b (n d) w h -> b n d w h', n=3)
        isp_img_split = isp_img.unsqueeze(2) 

        split_raw = einsum('b n c w h, b n d w h -> b n d w h', isp_img_split, split_conf) 
        split_raw = rearrange(split_raw, 'b n d w h -> b (n d) w h')

        FineTune_vector = self.finetune_merge_1(torch.cat((split_raw,split_vector),1))
        FineTune_vector = self.LReLU(self.body_finetune_1(FineTune_vector)) 
        finetune_raw = split_raw + FineTune_vector

        
        for i in range(finetune_num):
            FineTune_vector = self.finetune_merge_2(torch.cat((finetune_raw,FineTune_vector),1)) 
            FineTune_vector = self.LReLU(self.body_finetune_2(FineTune_vector))
            finetune_raw = finetune_raw + FineTune_vector
           

        rgb_fea = self.body_ISP(finetune_raw) 
        rgb_fea = self.conv_ISP_2(rgb_fea)
        rgb_fea = rgb_fea + finetune_raw
        bic_isp = F.interpolate(isp_img, scale_factor=4, mode='bicubic', align_corners=False)
        
        out = self.upsampler(rgb_fea) + bic_isp
        
        return out
    











        
        









