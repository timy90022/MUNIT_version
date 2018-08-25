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


out_path = './merge_folder/'
path = './convert_folder_01/'
# csvfile = path+"metadata_tmp.csv"
size = 128

if not os.path.exists(out_path):
    os.mkdir(out_path)


all_image = sorted(glob.glob(path + '/*.*'))

width = 1920
height = 960

merge_image = np.zeros((height,width,3),dtype=np.uint8)
# for i in range(len(all_image)):
#     im = cv2.imread(all_image[i])
#     print(all_image[i])
#     print(im.shape)
#     b, g, r = cv2.split(im)
#     im = cv2.merge([r, g, b])
index = 0
for w in range(0,height,size):
    for h in range(0,width,size):
        print(w,h)
        tmp = cv2.imread(all_image[index])
        b, g, r = cv2.split(tmp)
        im = cv2.merge([r, g, b])
        try:
            merge_image[w:w + size,h:h + size, :] = im
            index += 1
        except:
            merge_image[w:w+size,h:, :] = im
            index += 1
        print(index,len(all_image))
        if (index==len(all_image)):
            break
    if (index==len(all_image)):
        break


Image.fromarray(merge_image, 'RGB').save(out_path + 'test1.jpg')
