import math
import cv2
import numpy as np
import torch
import lpips

#--------------Y channel---------------
def calc_psnr(img1, img2):
    img1_array = img1.numpy()
    img2_array = img2.numpy()
    img1_array = np.transpose(img1_array, (1,2,0))
    img2_array = np.transpose(img2_array, (1,2,0))
    img1_array = np.where(img1_array >= 0., img1_array, 0.)
    img2_array = np.where(img2_array >= 0., img2_array, 0.)
    img1_array = np.where(img1_array <= 1., img1_array, 1.)
    img2_array = np.where(img2_array <= 1., img2_array, 1.)
    
    diff = (img1_array - img2_array)

    diff[:,:,0] = diff[:,:,0] * 65.738 / 256.0
    diff[:,:,1] = diff[:,:,1] * 129.057 / 256.0
    diff[:,:,2] = diff[:,:,2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))

    return -10 * math.log10(mse)

def ssim(img1, img2):
    
    
    C1 = (0.01*255 )**2
    C2 = (0.03*255 )**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_ssim(img1,img2):
    img1 = img1.numpy()
    img2 = img2.numpy()
    img1 = np.transpose(img1, (1,2,0))
    img2 = np.transpose(img2, (1,2,0))
    img1 = np.where(img1 >= 0., img1, 0.)
    img2 = np.where(img2 >= 0., img2, 0.)
    img1 = np.where(img1 <= 1., img1, 1.)
    img2 = np.where(img2 <= 1., img2, 1.)
    img1 = img1*255.0
    img2 = img2*255.0
    
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    return ssim(img1_y, img2_y)


lpips_fun = lpips.LPIPS(net='vgg').cuda()#.to(torch.device("cpu"))
def calc_lpips(img1,img2,function):
    with torch.no_grad():
        fid_value=function(img1,img2)
    return fid_value.cpu().detach().item()


class L1_loss(torch.nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
