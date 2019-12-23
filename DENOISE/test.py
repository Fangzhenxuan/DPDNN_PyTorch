# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:07:10 2019

@author: 31723
"""

import torch
from PIL import Image 
import numpy as np
from torchvision import transforms as T
from utils import add_noise
import math
from model import DPDNN
import torch.nn as nn
from config import opt

i = 1
label_img = './Set12/%.2d.png'%i


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return mse*255*255, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


transform1 = T.ToTensor()
transform2 = T.ToPILImage()

with torch.no_grad():
    net = DPDNN()
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(opt.load_model_path))

    img = Image.open(label_img)
    # img.show()
    label = np.array(img).astype(np.float32)   # label:0~255
    img_H = img.size[0]
    img_W = img.size[1]
    img = transform1(img)

    for j in range(10):
        img_noise = add_noise(img, opt.noise_level).resize_(1, 1, img_H, img_W)

        output = net(img_noise)
        output = output.cpu()
        output = output.resize_(img_H, img_W)
        output = torch.clamp(output, min=0, max=1)
        output = transform2(output)

        # output.show()
        output.save('./output/sigma%d/%d.png'%(opt.noise_level, i))

        img_noise = transform2(img_noise.resize_(img_H, img_W))
        # img_noise.show()
        # img_noise.save('./output/sigma%d/%d_noise.png'%(opt.noise_level, i))
        output = np.array(output)   # output:0~255

        print(i, PSNR(output, label))










