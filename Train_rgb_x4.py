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
from network import VSC_rgb as VSC
from Our_dataloader_rgb import train_GetData,val_GetData
from evaluation_metrics import calc_psnr,calc_ssim,L1_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flag", type=int, default=1)
    parser.add_argument("--Epoch", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--crop_size", type=int, default=48) 
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers.')
    parser.add_argument("--LR", type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--use_L2', action='store_true', default=False, help='Use L2 loss instead of L1 loss.')
    parser.add_argument('--Lambda_1', type=float, default=1., help='rgb loss weight.')
    parser.add_argument('--Lambda_2', type=float, default=0., help='raw loss weight.')
    parser.add_argument('--crop_border_flag', action='store_true', default=True, help='Use pretrain model.')
    parser.add_argument('--crop_border', type=int, default=4, help='crop_border.')
    parser.add_argument('--pretrain', action='store_true', default=True, help='Use pretrain model.')
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
    test_list = ['Set5','Set14','BSD100','Urban100']
    for i in range(len(test_list)):
        PSNR_all_x4 = [] 
        SSIM_all_x4 = [] 
        for iter, (hr_input_x4, lr_input_x4) in enumerate(valloader[i]): #,Lr_input_hf
            hr_input_x4 = hr_input_x4.to(DEVICE).float()
            lr_input_x4 = lr_input_x4.to(DEVICE).float()
            lr_x4_raw = lr_x4_raw.to(DEVICE).float()

            with torch.no_grad():
                SR_rgb_x4 = net.forward(lr_input_x4,2)

            vutils.save_image(SR_rgb_x4, './save_model/'+cfg.experiment_index+'/'+test_list[i]+'_SR_rgb_x4'+'_'+str(iter)+'_'+cfg.experiment_index+'.png')
            

            if cfg.crop_border_flag:
                PSNR_our_x4 = calc_psnr(hr_input_x4.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border],SR_rgb_x4.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border])
                SSIM_our_x4 = calc_ssim(hr_input_x4.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border],SR_rgb_x4.cpu().detach()[0, :, cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border])
            
            PSNR_all_x4.append(PSNR_our_x4)
            SSIM_all_x4.append(SSIM_our_x4)
            print(test_list[i]+' / '+str(iter+1))
        
        PSNR_all_value_x4, SSIM_all_value_x4 = sum(PSNR_all_x4)/len(PSNR_all_x4), sum(SSIM_all_x4)/len(SSIM_all_x4)
       
        with open('./log/Log_metric_'+cfg.experiment_index+'.txt',"a") as f2:
                f2.write(test_list[i]+': '+str(epoch)+' | '+' | '+Data_path+' | '+'PSNR_x2: '+str(0.000)+' | '+'SSIM_x2: '+str(0.000)+' | '+'PSNR_x4: '+str(PSNR_all_value_x4)+' | '+'SSIM_x4: '+str(SSIM_all_value_x4)+'\n')
                f2.write('\n')
   





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
        
        scheduler_G = lr_scheduler.MultiStepLR(optimizer_G,milestones=[50000,150000,250000,500000],gamma = 0.5)

        if cfg.use_L2:
            criterion_pixelwise = nn.MSELoss()
        else:
            criterion_pixelwise = L1_loss()

            
        trainset = train_GetData('./DataSet/DF2K',cfg.crop_size)
        trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

        valset_set5 = val_GetData('./DataSet/Set5')
        valloader_set5 = DataLoader(valset_set5, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

        valset_set14 = val_GetData('./DataSet/Set14')
        valloader_set14 = DataLoader(valset_set14, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

        valset_BSD100 = val_GetData('./DataSet/BSD100')
        valloader_BSD100 = DataLoader(valset_BSD100, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

        valset_Urban100 = val_GetData('./DataSet/Urban100')
        valloader_Urban100 = DataLoader(valset_Urban100, batch_size=1, shuffle=False, num_workers=cfg.num_workers)


        count_iter=0
        for epoch_count in range(cfg.Epoch):

            for iter, (hr_crop,lr_x2_crop,lr_x4_crop) in enumerate(trainloader):
                count_iter += 1
                hr_crop = hr_crop.to(DEVICE).float()

                lr_x4_crop = lr_x4_crop.to(DEVICE).float()
                
                SR_rgb_x4 = net.forward(lr_x4_crop)
                loss_pixel_rgb_x4 = criterion_pixelwise(hr_crop, SR_rgb_x4)

                All_loss = cfg.Lambda_1*loss_pixel_rgb_x4
                    
                optimizer_G.zero_grad()
                
                All_loss.backward()
                optimizer_G.step()
                scheduler_G.step()

            
                print('current epoch: %d | learn_rate: %4f | current count: %d | RGB_pixel_loss_x4: %4f | RAW_pixel_loss_x4: %4f | RGB_pixel_loss_x2: %4f | RAW_pixel_loss_x2: %4f'%(epoch_count+1,optimizer_G.param_groups[0]['lr'],count_iter,cfg.Lambda_1 * loss_pixel_rgb_x4.detach().item(),0. , 0.,0.))


                if epoch_count+1 <= 0:
                    if (count_iter) % 5000 == 0:
                        torch.save(net,os.path.join('./save_model',cfg.experiment_index,'VSCNet_'+str(count_iter)+'_'+cfg.experiment_index+'.pkl'))
                        evolution(cfg,'DIV2K',net,int(count_iter),[valloader_set5,valloader_set14,valloader_BSD100,valloader_Urban100])
                        net.train()
                        torch.cuda.empty_cache() 
                else: 
                    if (count_iter) % 1000 == 0:
                        torch.save(net,os.path.join('./save_model',cfg.experiment_index,'VSCNet_'+str(count_iter)+'_'+cfg.experiment_index+'.pkl'))
                        evolution(cfg,'DIV2K',net,int(count_iter),[valloader_set5,valloader_set14,valloader_BSD100,valloader_Urban100])
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

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cfg,DEVICE)
