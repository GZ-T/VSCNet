import os
import argparse
import random
import numpy as np
import time
import torchvision 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torchvision import utils as vutils
import torch.nn.functional as F
from torch.optim import lr_scheduler
from network import VSC_raw as VSC
from Our_dataloader_raw import train_GetData,val_GetData
from evaluation_metrics import calc_psnr,calc_ssim,L1_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flag", type=int, default=1)
    parser.add_argument("--Epoch", type=int, default=7200)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers.')
    parser.add_argument("--LR", type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--use_L2', action='store_true', default=False, help='Use L2 loss instead of L1 loss.')
    parser.add_argument('--Lambda_1', type=float, default=1., help='rgb loss weight.')
    parser.add_argument('--Lambda_2', type=float, default=0.01, help='raw loss weight.')
    parser.add_argument('--crop_border_flag', action='store_true', default=True, help='Use pretrain model.')
    parser.add_argument('--crop_border', type=int, default=4, help='crop_border.')
    parser.add_argument('--pretrain', action='store_true', default=False, help='Use pretrain model.')
    parser.add_argument('--pretrain_model_path', type=str, default='', help='pretrain model path.')
    parser.add_argument('--experiment_index', type=str, default='default', help='the experiment_index.')
    parser.add_argument('--rondom_seed', type=int, default=1, help='rondom seed.') 
    return parser.parse_args()

def setup_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)# 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evolution(cfg,Data_path,net,epoch,valloader):
    net.eval()
    PSNR_all = [] 
    SSIM_all = [] 

    for iter, (gt_rgb_image,raw_image,isp_image) in enumerate(valloader): #,Lr_input_hf
        gt_rgb_image = gt_rgb_image.to(DEVICE).float()
        raw_image = raw_image.to(DEVICE).float()
        isp_image = isp_image.to(DEVICE).float()

        with torch.no_grad():
            SR_rgb,SR_raw = net.forward(raw_image,isp_image)

        vutils.save_image(SR_rgb, './save_model/'+cfg.experiment_index+'/SR_rgb'+'_'+str(iter)+'_'+cfg.experiment_index+'.png')
        vutils.save_image(SR_raw, './save_model/'+cfg.experiment_index+'/SR_raw'+'_'+str(iter)+'_'+cfg.experiment_index+'.png')



        if cfg.crop_border_flag:
            PSNR_our = calc_psnr(gt_rgb_image.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border],SR_rgb.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border])
            SSIM_our = calc_ssim(gt_rgb_image.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border],SR_rgb.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border])
        else:
            PSNR_our = calc_psnr(gt_rgb_image.cpu().detach()[0],SR_rgb.cpu().detach()[0])
            SSIM_our = calc_ssim(gt_rgb_image.cpu().detach()[0],SR_rgb.cpu().detach()[0])



        PSNR_all.append(PSNR_our)
        SSIM_all.append(SSIM_our)
        print(str(iter+1)+' / '+str(15))
        
    PSNR_all_value, SSIM_all_value = sum(PSNR_all)/len(PSNR_all), sum(SSIM_all)/len(SSIM_all)
    with open('./log/Log_metric_'+cfg.experiment_index+'.txt',"a") as f2:
            f2.write('RAWSR_mini Count iter: '+str(epoch)+' | '+' | '+Data_path+' | '+'PSNR: '+str(PSNR_all_value)+' | '+'SSIM: '+str(SSIM_all_value)+'\n')
            f2.write('\n')
    return PSNR_all_value, SSIM_all_value  




def main(cfg,DEVICE):
    # define network
    net = VSC.VSCNet().to(DEVICE)
    if cfg.pretrain:
        net = torch.load(cfg.pretrain_model_path)
        net.eval()
        print('Load pretrain teacher model complete.') 

    if cfg.train_flag == 1:
        with open('./log/Log_trainset_'+cfg.experiment_index+'.txt',"a") as f:
            f.write('Total epochs: '+str(cfg.Epoch)+'\n')
            f.write('Batch size: '+str(cfg.batch_size)+'\n')
            f.write('Crop size: '+str(cfg.crop_size)+'\n')
            f.write('Initial learning rate: '+str(cfg.LR)+'\n')
            f.write('RGB loss weight: '+str(cfg.Lambda_1)+'\n')
            f.write('RAW loss weight: '+str(cfg.Lambda_2)+'\n')
            f.write('Use pretrain model: '+str(cfg.pretrain)+'\n')
            f.write('random seed: '+str(cfg.rondom_seed)+'\n')

        begin_time = time.time()
        net.train()
        optimizer_G = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=cfg.LR,
            betas=(0.9, 0.99)
            )
        
        scheduler_G = lr_scheduler.MultiStepLR(optimizer_G,milestones=[10000,50000,150000,250000,500000],gamma = 0.5)

        if cfg.use_L2:
            criterion_pixelwise = nn.MSELoss()
        else:
            criterion_pixelwise = L1_loss()

            
        trainset = train_GetData('./DataSet/RawSR/train',cfg.crop_size)
        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

        valset = val_GetData('./DataSet/RawSR/test')
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)



        count_iter=0
        for epoch_count in range(cfg.Epoch):

            for iter, (gt_rgb_image,gt_raw_image,raw_image,isp_image) in enumerate(trainloader):
                count_iter += 1

                gt_rgb_image = gt_rgb_image.to(DEVICE).float()
                gt_raw_image = gt_raw_image.to(DEVICE).float()
                raw_image = raw_image.to(DEVICE).float()
                isp_image = isp_image.to(DEVICE).float()
                
                SR_rgb,SR_raw = net.forward(raw_image,isp_image)

                loss_pixel_rgb = criterion_pixelwise(gt_rgb_image, SR_rgb)
                loss_pixel_raw = criterion_pixelwise(gt_raw_image, SR_raw)
                All_loss = cfg.Lambda_1*loss_pixel_rgb + cfg.Lambda_2*loss_pixel_raw
                    
  
                optimizer_G.zero_grad()

                All_loss.backward()
                optimizer_G.step()
                scheduler_G.step()

                print('current epoch: %d | learn_rate: %4f | current count: %d | RGB_pixel_loss: %4f | RAW_pixel_loss: %4f'%(epoch_count+1,optimizer_G.param_groups[0]['lr'],count_iter,cfg.Lambda_1 * loss_pixel_rgb.detach().item(),cfg.Lambda_2 * loss_pixel_raw.detach().item()))


                if epoch_count+1 <= 3000:
                    if (count_iter) % 5000 == 0:
                        torch.save(net,os.path.join('./save_model',cfg.experiment_index,'VSCNet_'+str(count_iter)+'_'+cfg.experiment_index+'.pkl'))
                        PSNR_value, SSIM_value = evolution(cfg,'RAWSR',net,int(count_iter),valloader)
                        net.train()
                        torch.cuda.empty_cache() 
                else: 
                    if (count_iter) % 1000 == 0:
                        torch.save(net,os.path.join('./save_model',cfg.experiment_index,'VSCNet_'+str(count_iter)+'_'+cfg.experiment_index+'.pkl'))
                        PSNR_value, SSIM_value = evolution(cfg,'RAWSR',net,int(count_iter),valloader)
                        net.train()
                        torch.cuda.empty_cache() 

            
        print('run time: ',time.time()-begin_time)        




if __name__ == "__main__":
    cfg = parse_args()
    setup_seed(cfg.rondom_seed)
    print(torch.__version__)
    print(torchvision.__version__)
    print (os.getcwd()) 
    print (os.path.abspath('..')) #Get the parent directory of the current working directory

    if not os.path.exists(os.path.join('./save_model',cfg.experiment_index)):
        os.makedirs(os.path.join('./save_model',cfg.experiment_index))

    if not os.path.exists(os.path.join('./log')):
        os.makedirs(os.path.join('./log'))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cfg,DEVICE)
