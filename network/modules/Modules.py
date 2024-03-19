import math
import torch
import torch.nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from collections import OrderedDict
from torch.nn import functional as F


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, 1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn
      
class VAB(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.atten_branch = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = nn.LayerNorm(d_model)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x       
    






    
        










    

        





    

        








    


        



    


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32,oc=64, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, oc, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
    
    def forward2(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf=128, gc=32,oc=128):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, oc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, oc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, oc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m
    
    def forward_test(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m,m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused    

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Encoder_teacher(nn.Module):
    def __init__(self, in_nc=3, out_nc=128, nf=64, nb=3, gc=32):
        super(Encoder_teacher, self).__init__()
        self.encoder = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(in_nc, 16, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # input size. (3) x 64 x 64
            nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 32 x 32
            nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, out_nc, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 16 x 16
            # nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=False),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), bias=False),
            # nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x_encode = self.encoder(x)
        return x_encode
    

class Encoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=64, nf=64, nb=3, gc=32):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(in_nc, 16, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # input size. (3) x 64 x 64
            nn.Conv2d(16, 32, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 32 x 32
            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, out_nc, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 16 x 16
            # nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=False),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), bias=False),
            # nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x_encode = self.encoder(x)
        return x_encode
    

# class Reconstructor(nn.Module):
#     def __init__(self, n_resblocks, conv=common.default_conv):
#         super(Reconstructor, self).__init__()

#         n_resblocks = n_resblocks
#         n_feats = 128
#         kernel_size = 3 
#         scale = 4
#         act = nn.ReLU(True)

#         # define body module
#         m_body = [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act,
#             ) for _ in range(n_resblocks)
#         ]
#         m_body.append(conv(n_feats, n_feats, kernel_size))

#         # define tail module
#         m_tail = [
#             common.Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, 3, kernel_size)
#         ]

#         # self.head = nn.Sequential(*m_head)
#         self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)

#     def forward(self, x):
#         res = self.body(x)
#         res += x

#         x = self.tail(res)
#         return x 


class Reconstructor(nn.Module):
    def __init__(self, in_nc=3, nf=128, num_modules=3, out_nc=3, upscale=4):
        super(Reconstructor, self).__init__()

        # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        # self.B4 = RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0


    def forward(self, x):
        # out_fea = self.fea_conv(input)
        out_B1 = self.B1(x)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        # out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3], dim=1))
        # out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + x

        output = self.upsampler(out_lr)

        return output
    

class memory_update(nn.Module):
    '''memory_update module'''

    def __init__(self, nf=128, gc=32, oc=128):
        super(memory_update, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, oc)
        self.RDB2 = ResidualDenseBlock_5C(nf//2, gc, oc//2)
        self.AT = ESA(oc//2, nn.Conv2d)
        # self.RDB3 = ResidualDenseBlock_5C(nf//2, gc, oc//2)
        self.conv = nn.Conv2d(nf, 64, 3, 1, 1, bias=True)
        

    def forward(self, x,memory_bank):
        # x_row = torch.chunk(x,cfg.memory_row,-2)
        # x_column_list = []
        # for i in range(cfg.memory_row):
        #     x_column_list.append(torch.chunk(x_row[i],cfg.memory_column,-1)) # the size of x_column_list is (9,16,B,C,w,h)
        memory_diff = 2*x - memory_bank
        memory_new = self.RDB1(torch.cat((memory_diff,memory_bank),1))
        memory_new = self.conv(memory_new)
        memory_new = self.RDB2(memory_new)
        # memory_new = self.RDB3(memory_new)
        memory_new = self.AT(memory_new)
        updated_memory = memory_new

        return updated_memory
    
    def forward_test(self, x,memory_bank):
        # x_row = torch.chunk(x,cfg.memory_row,-2)
        # x_column_list = []
        # for i in range(cfg.memory_row):
        #     x_column_list.append(torch.chunk(x_row[i],cfg.memory_column,-1)) # the size of x_column_list is (9,16,B,C,w,h)
        memory_diff = 2*x - memory_bank
        memory_new = self.RDB1(torch.cat((memory_diff,memory_bank),1))
        memory_new = self.conv(memory_new)
        memory_new = self.RDB2(memory_new)
        # memory_new = self.RDB3(memory_new)
        memory_new,AT_mask = self.AT.forward_test(memory_new)
        updated_memory = memory_new
        return updated_memory,AT_mask
        
    
    def forward_without_MemoryRes(self, x,memory_bank):
        # x_row = torch.chunk(x,cfg.memory_row,-2)
        # x_column_list = []
        # for i in range(cfg.memory_row):
        #     x_column_list.append(torch.chunk(x_row[i],cfg.memory_column,-1)) # the size of x_column_list is (9,16,B,C,w,h)
        # memory_diff = 2*x - memory_bank
        memory_new = self.RDB1(torch.cat((x,memory_bank),1)) 
        memory_new = self.conv(memory_new) 
        memory_new = self.RDB2(memory_new)
        # memory_new = self.RDB3(memory_new)
        # updated_memory = x+memory_new
        return memory_new 
    

class Flow_correction_block(nn.Module):
    '''Converge previous memory'''

    def __init__(self, nf=64, gc=32,oc=64):
        super(Flow_correction_block, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.RDB1 = ResidualDenseBlock_5C(nf*2, gc, oc*2)
        self.conv3 = nn.Conv2d(128, 2, 3, 1, 1, bias=False)
        # self.AT = ESA(oc//2, nn.Conv2d)
        # self.RDB2 = ResidualDenseBlock_5C(nf, gc, oc)
        # self.RDB3 = ResidualDenseBlock_5C(nf, gc, oc)

    def forward(self, Res_fea, coarse_flow):
        coarse_flow_fea = self.conv1(coarse_flow)
        Res_fea = self.conv2(Res_fea)
        flow_res_fea = self.RDB1.forward2(torch.cat((coarse_flow_fea,Res_fea),1))
        # converge_fea = F.interpolate(converge_fea, scale_factor=2, mode='bicubic', align_corners=False)
        flow_res = self.conv3(flow_res_fea)
        # out = self.RDB3(out)
        return flow_res+coarse_flow





def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)
    
class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        
        if load_path:
            state_dict = OrderedDict()
            for k, v in torch.load(load_path).items():
                k = k.replace('moduleBasic', 'basic_module')
                state_dict[k] = v
            self.load_state_dict(state_dict)

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(
                0,
                F.avg_pool2d(
                    input=ref[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.insert(
                0,
                F.avg_pool2d(
                    input=supp[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))

        flow = ref[0].new_zeros([
            ref[0].size(0), 2,
            int(math.floor(ref[0].size(2) / 2.0)),
            int(math.floor(ref[0].size(3) / 2.0))
        ])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(
                input=flow,
                scale_factor=2,
                mode='bilinear',
                align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level],
                    upsampled_flow.permute(0, 2, 3, 1),
                    interp_mode='bilinear',
                    padding_mode='border'), upsampled_flow
            ], 1)) + upsampled_flow
        return flow

    def forward(self, ref, supp):
    #         print(ref.size())
    #         print(supp.size())
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(
            input=ref,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)

        flow = F.interpolate(
            input=self.process(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow

def memory_search(x,y):
    """
        calculate cosine similarity to search most relate memory information
        x: the current input frame, which size is (B,3,180,320) --> REDS4 dataset 
        y: the previous input frame, which size is (B,3,180,320) --> REDS4 dataset
    """
    memory_row = 9
    memory_column = 16
    calculate_cos = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
    # offset = 1

    x_gray = 0.299*x[:,0,:,:]+0.587*x[:,1,:,:]+0.114*x[:,2,:,:] # (B,H,W)
    y_gray = 0.299*y[:,0,:,:]+0.587*y[:,1,:,:]+0.114*y[:,2,:,:] # (B,H,W)

    x_row = torch.chunk(x_gray,memory_row,-2)
    x_column_list = []
    for i in range(memory_row):
        x_column_list.append(torch.chunk(x_row[i],memory_column,-1)) # the size of x_column_list is (B,9,16)

    y_row = torch.chunk(y_gray,memory_row,-2)
    y_column_list = []
    for i in range(memory_row):
        y_column_list.append(torch.chunk(y_row[i],memory_column,-1)) # the size of y_column_list is (B,9,16)

    b,h,w = x_column_list[0][0].shape
    search_result = np.zeros((b,h,w,2))

    # search_area=[]
    for i in range(memory_row):
        for j in range(memory_column):
            # Define search_area
            if i==0 or i==memory_row-1 or j==0 or j==memory_column-1:
                search_area=[]
                if memory_row-1>=i-1>=0 and memory_column-1>=j-1>=0:
                    search_area.append([i-1,j-1])
                if memory_row-1>=i-1>=0 and memory_column-1>=j>=0:
                    search_area.append([i-1,j])
                if memory_row-1>=i-1>=0 and memory_column-1>=j+1>=0:
                    search_area.append([i-1,j+1])
                if memory_row-1>=i>=0 and memory_column-1>=j-1>=0:
                    search_area.append([i,j-1])

                search_area.append([i,j])

                if memory_row-1>=i>=0 and memory_column-1>=j+1>=0:
                    search_area.append([i,j+1])
                if memory_row-1>=i+1>=0 and memory_column-1>=j-1>=0:
                    search_area.append([i+1,j-1])
                if memory_row-1>=i+1>=0 and memory_column-1>=j>=0:
                    search_area.append([i+1,j])
                if memory_row-1>=i+1>=0 and memory_column-1>=j+1>=0:
                    search_area.append([i+1,j+1])
            else:
                search_area=[[i-1,j-1],[i-1,j],[i-1,j+1],
                             [i,j-1],  [i,j],  [i,j+1],
                             [i+1,j-1],[i+1,j],[i+1,j+1]]
                
            x_flatten = x_column_list[i][j].reshape(b,h*w)  
            cosine_similarity_list = []  
            for index in range(len(search_area)):
                y_flatten = y_column_list[search_area[index][0]][search_area[index][1]].reshape(b,h*w)
                cs_value = calculate_cos(x_flatten, y_flatten) # [b]
                cosine_similarity_list.append(cs_value.tolist()) # [len(search_area),b]

            cosine_similarity_array = np.array(cosine_similarity_list)

            # memory_index_list = []
            # for index in range(len(cosine_similarity_list)):
            for index in range(b):
                single_cs_list = list(cosine_similarity_array[:,index])
                max_index = single_cs_list.index(max(single_cs_list))
                # memory_index_list.append(search_area[max_index])
                search_result[index,i,j,0] = search_area[max_index][0]
                search_result[index,i,j,1] = search_area[max_index][1]

    return search_result








                



                









