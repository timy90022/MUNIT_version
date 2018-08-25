import numpy as np
import os
import json
import cv2
# import  angle as angle
# import transfer as trans
# import visiual as visiual
from PIL import Image, ImageDraw
import pandas as pd
# import Equirec2Perspec as E2P
import torch
# import poc_data as poc
from image_match.goldberg import ImageSignature
from skimage.feature import register_translation
import random
import decimal
import csv
import glob


path = '/home/timy90022/VR/inpainting_super_resolution/55/'
out_path = './cut_folder/'
# csvfile = path+"metadata_tmp.csv"
size = 128

if not os.path.exists(out_path):
    os.mkdir(out_path)


all_image = sorted(glob.glob(path + 'images'+ '/*.*'))

# print(all_image)

for i in range(1):
    im = cv2.imread(all_image[i])
    print(all_image[i])
    print(im.shape)
    b, g, r = cv2.split(im)
    im = cv2.merge([r, g, b])
    for w in range(0,im.shape[0],size):
        for h in range(0,im.shape[1],size):
            print(i,w,h)
            crop_image = im[w:w + size,h:h + size, :]
            Image.fromarray(crop_image, 'RGB').save(out_path + '%04d_%04d.jpg'%(w,h))
