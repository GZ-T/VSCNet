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


def pixelshuffle_single(in_channels, out_channels, upscale_factor=2):
    upconv1 = nn.Conv2d(in_channels, 56, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    upconv2 = nn.Conv2d(56, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])


class VSCNet(nn.Module):
   
    def __init__(self, num_feat=48, num_blocks=30):
        super().__init__()

        self.pixel_unshuffle_layer = nn.PixelUnshuffle(2)
        self.pixel_shuffle_layer = nn.PixelShuffle(2)
        self.upchannel = nn.Conv2d(4, num_feat, 3, 1, 1)
        self.body_split = make_layer(modules.VAB, 3, num_feat, 64)
        self.sigmod = nn.Sigmoid()

        self.color_decouple = nn.Conv2d(num_feat, 16, 3, 1, 1, groups=4)
        self.color_global = nn.Conv2d(num_feat, 16, 1, 1, 0)
        self.CD_body_merge = make_layer(modules.VAB, 3, 16, 64)
        self.CG_body_merge = make_layer(modules.VAB, 3, 16, 64)

        self.body_finetune_1 = make_layer(modules.VAB, 3, 4+4, 64)
        self.finetune_merge_1 = nn.Conv2d(8, 4, 3, 1, 1)
        self.body_finetune_2 = make_layer(modules.VAB, 6, 4+4, 64)
        self.finetune_merge_2 = nn.Conv2d(8, 4, 3, 1, 1)
        self.LReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        

        self.conv_ISP_1 = nn.Conv2d(4+3, num_feat, 3, 1, 1)
        self.body_ISP = make_layer(modules.VAB, 6, num_feat, 64)
        self.conv_ISP_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=2)

        self.upsampler = pixelshuffle_single(num_feat, 3, upscale_factor=2)



    def forward(self, raw_img, isp_img, finetune_num=2):
        """

        Args:
            raw_img (Tensor): Input raw images with shape (n, 1, h, w)
            isp_img (Tensor): Input isp images with shape (n, 3, h, w)
            finetune_num (int): finetuen operation number
        Returns:
            Tensor: Output feature with shape (n, out_channels, h*2, w*2)
        """
        #-----------------SPLIT---------------------
        raw_img = self.pixel_unshuffle_layer(raw_img) 
        raw_fea = self.upchannel(raw_img) 
        split_vector = self.body_split(raw_fea) 


        CD_split_cof = self.color_decouple(split_vector)
        CD_split_cof = self.CD_body_merge(CD_split_cof) 
        CG_split_cof = self.color_global(split_vector)
        CG_split_cof = self.CG_body_merge(CG_split_cof) 
        

        split_vector = self.sigmod(CD_split_cof+CG_split_cof) 
        split_cof = rearrange(split_vector, 'b (n d) w h -> b n d w h', n=4) 
        raw_img = raw_img.unsqueeze(2) 
        

        split_raw = einsum('b n c w h, b n d w h -> b n d w h', raw_img, split_cof) 
        split_raw = rearrange(split_raw, 'b n d w h -> b (n d) w h') 
        split_raw = self.pixel_shuffle_layer(split_raw) 
        split_vector = self.pixel_shuffle_layer(split_vector) 

         
        #-----------------FINE-TUNE---------------------
        FineTune_vector = self.body_finetune_1(torch.cat((split_raw,split_vector),1)) 
        FineTune_vector = self.LReLU(self.finetune_merge_1(FineTune_vector)) 
        finetune_raw = split_raw + FineTune_vector


        
        for i in range(finetune_num):
            FineTune_vector = self.body_finetune_2(torch.cat((finetune_raw,FineTune_vector),1)) 
            FineTune_vector = self.LReLU(self.finetune_merge_2(FineTune_vector)) 
            finetune_raw = finetune_raw + FineTune_vector


        ISP_fea = self.conv_ISP_1(torch.cat((finetune_raw,isp_img),1))
        rgb_fea = self.body_ISP(ISP_fea) 
        rgb_fea = self.conv_ISP_2(rgb_fea)
        rgb_fea = rgb_fea + ISP_fea
        out = self.upsampler(rgb_fea)

        return out,F.pixel_shuffle(finetune_raw,2)








        
        









