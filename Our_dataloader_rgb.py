import glob
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import numpy as np 
from PIL import Image




class train_GetData(Dataset):
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
            im1 = np.fliplr(im1)
            im2 = np.fliplr(im2)
            im3 = np.fliplr(im3)

        angle = random.choice([0, 1, 2, 3])
        im1 = np.rot90(im1, angle)
        im2 = np.rot90(im2, angle)
        im3 = np.rot90(im3, angle)

        # have to copy before be called by transform function
        return im1.copy(), im2.copy(), im3.copy()
    
    

    def __getitem__(self, index):
        hr_input = np.array(Image.open(self.hr_img_list[index]))/255.
        lr_input_x2 = np.array(Image.open(os.path.join(self.dir, 'LR_x2', self.hr_img_list[index].split('/')[-1][:-4]+'x2.png')))/255.
        lr_input_x4 = np.array(Image.open(os.path.join(self.dir, 'LR_x4', self.hr_img_list[index].split('/')[-1][:-4]+'x4.png')))/255.


        hr_crop,lr_x2_crop,lr_x4_crop = self.random_crop(hr_input,lr_input_x2,lr_input_x4,self.crop_size)
        hr_crop,lr_x2_crop,lr_x4_crop = self.random_flip_and_rotate(hr_crop,lr_x2_crop,lr_x4_crop)


        hr_crop = self.transform(hr_crop)
        lr_x2_crop = self.transform(lr_x2_crop)
        lr_x4_crop = self.transform(lr_x4_crop)
        
        return hr_crop,lr_x2_crop,lr_x4_crop



class val_GetData(Dataset):
    def __init__(self, Dir,Is_Training=False):
        self.dir = Dir 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.is_training = Is_Training 
        self.hr_img_list = sorted(glob.glob(os.path.join(self.dir,'x4/HR', '*.png')))
        # self.hr_img_list = sorted(glob.glob(os.path.join(self.dir,'x2/HR', '*.png')))
                     
    def __len__(self):
        return len(self.hr_img_list)
    
    
    def __getitem__(self, index):
        hr_input_x4 = np.array(Image.open(self.hr_img_list[index]))/255.
        lr_input_x4 = np.array(Image.open(os.path.join(self.dir, 'x4/LR', self.hr_img_list[index].split('/')[-1][:-8]+'4_LR.png')))/255.
        if len(hr_input_x4.shape) == 2:
            # hr_input_x2 = hr_input_x2[:,:,np.newaxis]
            # hr_input_x2 = np.concatenate((hr_input_x2, hr_input_x2, hr_input_x2), axis=2)
            # lr_input_x2 = lr_input_x2[:,:,np.newaxis]
            # lr_input_x2 = np.concatenate((lr_input_x2, lr_input_x2, lr_input_x2), axis=2)
            hr_input_x4 = hr_input_x4[:,:,np.newaxis]
            hr_input_x4 = np.concatenate((hr_input_x4, hr_input_x4, hr_input_x4), axis=2)
            lr_input_x4 = lr_input_x4[:,:,np.newaxis]
            lr_input_x4 = np.concatenate((lr_input_x4, lr_input_x4, lr_input_x4), axis=2)



        
        # hr_input_x2 = self.transform(hr_input_x2)
        # lr_input_x2 = self.transform(lr_input_x2)

        hr_input_x4 = self.transform(hr_input_x4)
        lr_input_x4 = self.transform(lr_input_x4)
        
        return hr_input_x4, lr_input_x4




