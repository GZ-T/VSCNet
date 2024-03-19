import torch
import cv2  
import glob
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader 
import random
import numpy as np 
from PIL import Image
import h5py


class train_GetData(Dataset):
    def __init__(self, Dir,Crop_size,Is_Training=True):
        self.dir = Dir 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.crop_size = Crop_size
        self.is_training = Is_Training 
        self.ISP_img_list = sorted(glob.glob(os.path.join(self.dir,'ISP', '*.jpeg')))

    def __len__(self):
        return len(self.ISP_img_list)

    def random_crop(self,gt,raw,isp,size,scale=2):
        h, w = isp.shape[:-1]
        x = random.randint(0, w-size)
        y = random.randint(0, h-size)
        
        size_4 = size*scale
        x_4, y_4 = x*scale, y*scale

        isp_crop = isp[y:y+size, x:x+size].copy()
        raw_crop = raw[y:y+size, x:x+size].copy()
        gt_crop = gt[y_4:y_4+size_4, x_4:x_4+size_4].copy()

        return gt_crop,raw_crop,isp_crop
    
    def random_flip_and_rotate(self,im1,im2,im3): 
        if random.random() < 0.5:
            im1 = np.flipud(im1)
            im2 = np.flipud(im2)
            im3 = np.flipud(im3)

        if random.random() < 0.5:
            im1 = np.fliplr(im1)
            im2 = np.fliplr(im2)
            im3 = np.fliplr(im3)

        angle = random.choice([0, 1, 2, 3])
        im1 = np.rot90(im1, angle)
        im2 = np.rot90(im2, angle)
        im3 = np.rot90(im3, angle)

        # have to copy before be called by transform function
        return im1.copy(), im2.copy(), im3.copy()
    
    def bayerprocess(self,rgb):
        rgb = rgb.astype(np.float32)#/255.#/65535.
        bayer = np.copy(rgb[:, :, 0])
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return bayer

    def __getitem__(self, index):
        isp_input = np.array(Image.open(self.ISP_img_list[index]))
        raw_input = np.load(os.path.join(self.dir, 'TrainingSet', self.ISP_img_list[index].split('/')[-1][:-5]+'.npy'))
        gt = np.load(os.path.join(self.dir, 'GT', self.ISP_img_list[index].split('/')[-1][:-9]+'.npy'))

        gt_crop,raw_crop,isp_crop = self.random_crop(gt,raw_input,isp_input,self.crop_size)
        gt_crop,raw_crop,isp_crop = self.random_flip_and_rotate(gt_crop,raw_crop,isp_crop)

        gt_raw_crop = self.bayerprocess(gt_crop)

        gt_crop = self.transform(gt_crop)
        gt_raw_crop = self.transform(gt_raw_crop)
        raw_crop = self.transform(raw_crop)
        isp_crop = self.transform(isp_crop)
        
        return gt_crop,gt_raw_crop,raw_crop,isp_crop






class val_GetData(Dataset):
    def __init__(self, Dir,Is_Training=False):
        self.dir = Dir 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.is_training = Is_Training 
        self.ISP_img_list = sorted(glob.glob(os.path.join(self.dir,'ISP', '*.jpeg')))
                     
    def __len__(self):
        return len(self.ISP_img_list)
    
    def __getitem__(self, index):
        isp_input = np.array(Image.open(self.ISP_img_list[index]))
        raw_input = np.load(os.path.join(self.dir, 'TrainingSet', self.ISP_img_list[index].split('/')[-1][:-5]+'.npy'))
        gt = np.load(os.path.join(self.dir, 'GT', self.ISP_img_list[index].split('/')[-1][:-9]+'.npy'))

        # too large to test which need to crop
        # isp_input = isp_input[0:500, 0:500]
        # raw_input = raw_input[0:500, 0:500]
        # gt = gt[0:1000, 0:1000]
        
        gt = self.transform(gt)
        raw_input = self.transform(raw_input)
        isp_input = self.transform(isp_input)
        
        return gt, raw_input, isp_input
    

class train_GetData_div2k(Dataset):
    def __init__(self, Dir,FNames,Crop_Size,Is_Training = True):
        self.dir = Dir
        self.fnames = FNames
        # self.labels = Labels  
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.crop_size = Crop_Size 
        self.is_training = Is_Training

        h5f = h5py.File(self.dir, "r")
        self.hr = [v[:] for v in h5f["DIV2K_train_HR"].values()]
        # self.lr_x2 = [v[:] for v in h5f["DIV2K_train_LR_bicubic_X2"].values()]
        self.lr_x4 = [v[:] for v in h5f["DIV2K_train_LR_bicubic_X4"].values()]
        h5f.close()

                     
    def __len__(self):
        return len(self.fnames)

    def random_crop(self,hr, lr_x4, size):
        # h, w = lr_x2.shape[:-1]
        h, w = lr_x4.shape[:-1]
        # h, w, _ = np.shape(lr_x4)
        x = random.randint(0, w-size)
        y = random.randint(0, h-size)

        # size_2 = size*2
        # x_2, y_2 = x*2, y*2
        
        size_4 = size*4
        x_4, y_4 = x*4, y*4

        
        # crop_lr_x2 = lr_x2[y:y+size, x:x+size].copy()
        # crop_hr = hr[y_2:y_2+size_2, x_2:x_2+size_2].copy()
        crop_lr_x4 = lr_x4[y:y+size, x:x+size].copy()
        crop_hr = hr[y_4:y_4+size_4, x_4:x_4+size_4].copy()

        # return crop_hr, crop_lr_x2, crop_lr_x4
        return crop_hr, crop_lr_x4#crop_lr_x4


    def random_flip_and_rotate(self,im1, im2): 
        if random.random() < 0.5:
            im1 = np.flipud(im1)
            im2 = np.flipud(im2)
            # im3 = np.flipud(im3)

        if random.random() < 0.5:
            im1 = np.fliplr(im1)
            im2 = np.fliplr(im2)
            # im3 = np.fliplr(im3)

        angle = random.choice([0, 1, 2, 3])
        im1 = np.rot90(im1, angle)
        im2 = np.rot90(im2, angle)
        # im3 = np.rot90(im3, angle)

        # have to copy before be called by transform function
        return im1.copy(), im2.copy()#, im3.copy()
    

    def bayerprocess(self,rgb):
        rgb = rgb.astype(np.float32)#/255.#/65535.
        bayer = np.copy(rgb[:, :, 0])
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return bayer


    def __getitem__(self, index):

        x = self.lr_x4[index]/255.
        # x = self.lr_x2[index]
        y = self.hr[index]/255.
        
        y_crop, x_crop =  self.random_crop(y, x, self.crop_size)
        y_crop, x_crop =  self.random_flip_and_rotate(y_crop, x_crop)

        y_raw_crop = self.bayerprocess(y_crop)
        x_raw_crop = self.bayerprocess(x_crop)



        x_crop = self.transform(x_crop)
        y_crop = self.transform(y_crop)
        x_raw_crop = self.transform(x_raw_crop)
        y_raw_crop = self.transform(y_raw_crop)
        
        return x_crop,x_raw_crop,y_crop,y_raw_crop

class train_GetData_div2k_x2_4(Dataset):
    def __init__(self, Dir,Crop_size,Is_Training=True):
        self.dir = Dir 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.crop_size = Crop_size
        self.is_training = Is_Training 
        self.hr_img_list = sorted(glob.glob(os.path.join(self.dir,'HR','*.png')))

    def __len__(self):
        return len(self.hr_img_list)

    def random_crop(self,gt,lr_x2,lr_x4,size):
        h, w = lr_x4.shape[:-1]
        x = random.randint(0, w-size)
        y = random.randint(0, h-size)
        
        size_4 = size*4
        x_4, y_4 = x*4, y*4

        size_2 = size*2
        x_2, y_2 = x*2, y*2
        
        lr_x4_crop = lr_x4[y:y+size, x:x+size].copy()
        lr_x2_crop = lr_x2[y_2:y_2+size_2, x_2:x_2+size_2].copy()
        gt_crop = gt[y_4:y_4+size_4, x_4:x_4+size_4].copy()

        return gt_crop,lr_x2_crop,lr_x4_crop
    
    def random_flip_and_rotate(self,im1,im2,im3): 
        if random.random() < 0.5:
            im1 = np.flipud(im1)
            im2 = np.flipud(im2)
            im3 = np.flipud(im3)

        if random.random() < 0.5:
            im1 = np.fliplr(im1)
            im2 = np.fliplr(im2)
            im3 = np.fliplr(im3)

        angle = random.choice([0, 1, 2, 3])
        im1 = np.rot90(im1, angle)
        im2 = np.rot90(im2, angle)
        im3 = np.rot90(im3, angle)

        # have to copy before be called by transform function
        return im1.copy(), im2.copy(), im3.copy()
    
    def bayerprocess(self,rgb):
        rgb = rgb.astype(np.float32)#/255.#/65535.
        bayer = np.copy(rgb[:, :, 0])
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return bayer
    

    def __getitem__(self, index):
        hr_input = np.array(Image.open(self.hr_img_list[index]))/255.
        lr_input_x2 = np.array(Image.open(os.path.join(self.dir, 'LR_x2', self.hr_img_list[index].split('/')[-1][:-4]+'x2.png')))/255.
        lr_input_x4 = np.array(Image.open(os.path.join(self.dir, 'LR_x4', self.hr_img_list[index].split('/')[-1][:-4]+'x4.png')))/255.

        # h,w,_ = np.shape(hr_input.shape)
        # lr_input_x2 = hr_input.resize((h//2,))
        # lr_input = np.array(Image.open(os.path.join(self.dir, 'TrainingSet', self.hr_img_list[index].split('/')[-1][:-5]+'.npy')))

        hr_crop,lr_x2_crop,lr_x4_crop = self.random_crop(hr_input,lr_input_x2,lr_input_x4,self.crop_size)
        hr_crop,lr_x2_crop,lr_x4_crop = self.random_flip_and_rotate(hr_crop,lr_x2_crop,lr_x4_crop)

        hr_raw_crop = self.bayerprocess(hr_crop)
        lr_x2_raw_crop = self.bayerprocess(lr_x2_crop)
        lr_x4_raw_crop = self.bayerprocess(lr_x4_crop)

        hr_crop = self.transform(hr_crop)
        hr_raw_crop = self.transform(hr_raw_crop)
        
        lr_x2_crop = self.transform(lr_x2_crop)
        lr_x2_raw_crop = self.transform(lr_x2_raw_crop)

        lr_x4_crop = self.transform(lr_x4_crop)
        lr_x4_raw_crop = self.transform(lr_x4_raw_crop)
        
        
        return hr_crop,hr_raw_crop,lr_x2_crop,lr_x2_raw_crop,lr_x4_crop,lr_x4_raw_crop#,gt_raw_crop,raw_crop,isp_crop



class val_GetData_div2k(Dataset):
    def __init__(self, Dir,Is_Training=False):
        self.dir = Dir 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.is_training = Is_Training 
        # self.hr_img_list = sorted(glob.glob(os.path.join(self.dir,'HR_x4', '*.png')))
        self.hr_img_list = sorted(glob.glob(os.path.join(self.dir,'x2/HR', '*.png')))
                     
    def __len__(self):
        return len(self.hr_img_list)
    
    def bayerprocess(self,rgb):
        rgb = rgb.astype(np.float32)#/255.#/65535.
        bayer = np.copy(rgb[:, :, 0])
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return bayer
    
    def __getitem__(self, index):
        hr_input_x2 = np.array(Image.open(self.hr_img_list[index]))/255.
        lr_input_x2 = np.array(Image.open(os.path.join(self.dir, 'x2/LR', self.hr_img_list[index].split('/')[-1][:-8]+'2_LR.png')))/255.
        hr_input_x4 = np.array(Image.open(os.path.join(self.dir, 'x4/HR', self.hr_img_list[index].split('/')[-1][:-8]+'4_HR.png')))/255.
        lr_input_x4 = np.array(Image.open(os.path.join(self.dir, 'x4/LR', self.hr_img_list[index].split('/')[-1][:-8]+'4_LR.png')))/255.
        if len(hr_input_x2.shape) == 2:
            hr_input_x2 = hr_input_x2[:,:,np.newaxis]
            hr_input_x2 = np.concatenate((hr_input_x2, hr_input_x2, hr_input_x2), axis=2)
            lr_input_x2 = lr_input_x2[:,:,np.newaxis]
            lr_input_x2 = np.concatenate((lr_input_x2, lr_input_x2, lr_input_x2), axis=2)
            hr_input_x4 = hr_input_x4[:,:,np.newaxis]
            hr_input_x4 = np.concatenate((hr_input_x4, hr_input_x4, hr_input_x4), axis=2)
            lr_input_x4 = lr_input_x4[:,:,np.newaxis]
            lr_input_x4 = np.concatenate((lr_input_x4, lr_input_x4, lr_input_x4), axis=2)


        # h,w,c = lr_input_x4.shape
        # if h%2 !=0 :
        #     lr_input_x4=lr_input_x4[:-1,:,:]
        #     lr_input_x2=lr_input_x2[:-2,:,:]
        #     hr_input=hr_input[:-4,:,:]

        # if w%2 !=0 :
        #     lr_input_x4=lr_input_x4[:,:-1,:]
        #     lr_input_x2=lr_input_x2[:,:-2,:]
        #     hr_input=hr_input[:,:-4,:]

        
        

        lr_x2_raw_crop = self.bayerprocess(lr_input_x2)
        lr_x4_raw_crop = self.bayerprocess(lr_input_x4)
        
        hr_input_x2 = self.transform(hr_input_x2)

        lr_input_x2 = self.transform(lr_input_x2)
        lr_x2_raw_crop = self.transform(lr_x2_raw_crop)

        hr_input_x4 = self.transform(hr_input_x4)
        lr_input_x4 = self.transform(lr_input_x4)
        lr_x4_raw_crop = self.transform(lr_x4_raw_crop)
        
        return hr_input_x2, lr_input_x2, lr_x2_raw_crop, hr_input_x4, lr_input_x4, lr_x4_raw_crop




