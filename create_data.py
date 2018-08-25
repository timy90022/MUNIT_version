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


path = '../inpainting_super_resolution/55/'

dataset = True

if dataset:
    create_dataset=['high-2.jpg','high-3.jpg','high-4.jpg','high-5.jpg']
    # create_dataset=['high-2.jpg']

    out_number = 1
    move = 1000
    for i in create_dataset:

        im = cv2.imread(path + 'images/' + str(i))
        height, width, channels = im.shape
        b, g, r = cv2.split(im)
        clean_picture = cv2.merge([r, g, b])
        # array = cv2.line(clean_picture,(0,600),(5376,600),(255,0,0),3)
        # array = cv2.line(clean_picture,(2688,0),(2688,2688),(255,0,0),3)
        # array = cv2.line(clean_picture,(0,2050),(5376,2050),(255,0,0),3)

        array = clean_picture
        for times in range(1000):

        	z = random.randint(850,1800)
	        size = int(random.randint(128,256)/2)

        	crop_image = array[z - size:z + size,2688 - size:2688 + size, :]
	        Image.fromarray(crop_image, 'RGB').save('./high_data/'+ str(out_number) + '.jpg')


	        tmp = np.array(array[:,width-move:,:])
	        array[:,move:,:] = array[:,:width-move,:]
	        array[:,:move,:] = tmp



        	out_number +=1


    #     feature_name = np.where(data2['image'] == i, data2['name'], 0)
    #     feature_name = np.delete(feature_name, np.where(feature_name == 0)).reshape([-1, 1])

    #     test = np.array(df.index).astype(np.int32)
    #     ix = np.isin(test, feature_name)
    #     ix = np.where(ix)

    #     x = np.array(df['coordinates-x'][ix[0]]).reshape([-1, 1])
    #     y = np.array(df['coordinates-y'][ix[0]]).reshape([-1, 1])
    #     z = np.array(df['coordinates-z'][ix[0]]).reshape([-1, 1])
    #     picture_coordinate_tmp = np.hstack((x, y, z))
    #     # print(picture_coordinate_tmp)
    #     picture_coordinate = picture_coordinate_tmp.reshape([1,1,-1,3])

    #     out_points = trans.rotate_pointcloud(torch.FloatTensor(picture_coordinate),torch.FloatTensor(tmp))
    #     out_map = trans.reproject_pointcloud(torch.FloatTensor(out_points)).numpy().reshape([-1,2])

    #     plane_angle = out_map[:,0] + 180
    #     vertical_angle = (out_map[:,1]) + 90
    #     plane_angle = plane_angle * width / 360
    #     vertical_angle = vertical_angle * height / 180

    #     point_size = 2
    #     crop_size = 128
        
    #     # low_size = int(128/2)
    #     # high_size = int(256/2)
    #     # out_low_size = int(128/2)
    #     # out_high_size = int(512/2)

    #     for k in range(len(vertical_angle)):
    #         im = cv2.imread(path + 'images/' + str(i))
    #         height, width, channels = im.shape
    #         b, g, r = cv2.split(im)
    #         picture = cv2.merge([r, g, b])

    #         vertical = int(vertical_angle[k])
    #         plane = int(plane_angle[k])
    #         point_name = df.index[ix[0][k]]

    #         # draw red point
    #         # for n in range(point_size*2):
    #         #     for m in range(point_size*2):
    #         #         picture[vertical+n-point_size][plane+m-point_size] = [255, 0, 0]

    #         # get high resolution image 
    #         # try :
    #         #     high_image = picture[vertical - high_size:vertical + high_size,plane - high_size:plane + high_size, :]
    #         #     out_high_image = picture[vertical - out_high_size:vertical + out_high_size,plane - out_high_size:plane + out_high_size, :]
    #         #     res = cv2.resize(high_image, dsize=(low_size * 2, low_size * 2), interpolation=cv2.INTER_CUBIC)
    #         # except:
    #         #     continue



    #         # find image with same coordinate
    #         feature_name = np.where(data2['name'] == int(point_name), data2['image'], 0)
    #         feature_name = np.delete(feature_name, np.where(feature_name == 0)).reshape([-1, 1])
    #         picture_coordinate = picture_coordinate_tmp[k].reshape([1, 1, -1, 3])

    #         # distance in (high resolution image and coordinate)
    #         distance = np.sqrt(np.sum(np.power(picture_coordinate_tmp[k] - original_point, 2)))

    #         index = 0
    #         for name in feature_name:
    #             if "high" not in name[0] :
    #                 # get low resolution picture
    #                 im = cv2.imread(path + 'images/' + name[0])
    #                 height, width, channels = im.shape
    #                 b, g, r = cv2.split(im)
    #                 compare_picture = cv2.merge([r, g, b])

    #                 # get image information
    #                 tmp = data[0]['shots'][name[0]]['rotation']
    #                 t = data[0]['shots'][name[0]]['translation']
    #                 R = cv2.Rodrigues(np.array(tmp, dtype=float))[0]
    #                 tmp_point = np.array(-R.T.dot(t))
    #                 low_image_rt = np.hstack((np.array(R), np.array(t).reshape([3, 1]))).reshape([1, 3, 4])

    #                 # distance in (low resolution image and coordinate)
    #                 distance_tmp = np.sqrt(np.sum(np.power(picture_coordinate_tmp[k]-tmp_point,2)))

    #                 # reproject_point
    #                 out_points = trans.rotate_pointcloud(torch.FloatTensor(picture_coordinate), torch.FloatTensor(low_image_rt))
    #                 out_map_tmp = trans.reproject_pointcloud(torch.FloatTensor(out_points)).numpy().reshape([-1, 2])

    #                 # if(np.amax(np.absolute(out_map[k,:]-out_map_tmp))<10 or distance_tmp>0.3):
    #                 if( distance>0.4 and distance_tmp>0.3):
    #                     # print(distance)
    #                     # print(distance_tmp)
    #                     # print(distance / distance_tmp)
    #                     # magnification =   distance_tmp /distance

    #                     plane_angle_tmp = int(((out_map_tmp[:, 0] + 180)* width / 360)[0])
    #                     vertical_angle_tmp = int((((out_map_tmp[:, 1]) + 90)* height / 180)[0])

                        
    #                     compare_list = []
    #                     low_image = compare_picture[vertical_angle_tmp-int(crop_size/2):vertical_angle_tmp+int(crop_size/2),plane_angle_tmp-int(crop_size/2):plane_angle_tmp+int(crop_size/2),:]
                        
    #                     if (low_image.size!=0):
    #                         for large_ration in range(11):
    #                             enlarge = int(crop_size*(large_ration*0.2+2)/2)
    #                             high_image = picture[vertical - enlarge:vertical + enlarge,plane - enlarge:plane + enlarge, :]
    #                             try:
    #                                 far = gis.normalized_distance(gis.generate_signature(low_image), gis.generate_signature(high_image))
    #                                 compare_list.append(far)
    #                             except:
    #                                 if large_ration==0:
    #                                     break
    #                                 else:
    #                                     compare_list.append(1)

    #                         if len(compare_list)==0:
    #                             continue
    #                         # print(distance_tmp)
    #                         # print(compare_list.index(min(compare_list)))
    #                         enlarge = int((compare_list.index(min(compare_list))*0.2+2)*crop_size/2)
    #                         high_image = picture[vertical - enlarge:vertical + enlarge,plane - enlarge:plane + enlarge, :]
    #                         res = cv2.resize(high_image, dsize=(crop_size, crop_size), interpolation=cv2.INTER_CUBIC)

    #                         # assert False

    #                         # tmp_point    original_point
    #                         # proportion = np.sqrt(np.sum(np.power(np.vstack((original_point,tmp_point))-
    #                         #                                      picture_coordinate_tmp[k],2), axis=1, keepdims=True))
    #                         # proportion = np.round((proportion[0]/proportion[1]),decimals=1)
    #                         # print(proportion)

    #                         # get shift image
    #                         try:
    #                             shift, error, diffphase = register_translation(low_image, res)
    #                         except:
    #                             continue
    #                         y,x,z = shift
    #                         y = int(y)
    #                         x = int(x)

    #                         vertical_angle_tmp +=y
    #                         plane_angle_tmp +=x
    #                         out_low_image = compare_picture[vertical_angle_tmp-int(crop_size):vertical_angle_tmp+int(crop_size),plane_angle_tmp-int(crop_size):plane_angle_tmp+int(crop_size),:]
    #                         out_high_image = picture[vertical - enlarge*2:vertical + enlarge*2,plane - enlarge*2:plane + enlarge*2, :]

    #                         # calculate gis distance
    #                         out_low_image_test = gis.generate_signature(low_image)
    #                         out_high_image_test = gis.generate_signature(high_image)
    #                         far = gis.normalized_distance(out_low_image_test, out_high_image_test)

    #                         # concatenate image
    #                         tmp = cv2.resize(high_image, dsize=(crop_size, crop_size),interpolation=cv2.INTER_CUBIC)
    #                         out_image = np.concatenate((low_image, tmp), axis=1)

    #                         # compare image with hist
    #                         im1 = cv2.cvtColor(out_low_image, cv2.COLOR_RGB2GRAY)
    #                         hist1 = cv2.calcHist([im1], [0], None, [256], [0, 256])
    #                         im2 = cv2.cvtColor(out_high_image, cv2.COLOR_RGB2GRAY)
    #                         hist2 = cv2.calcHist([im2], [0], None, [256], [0, 256])
    #                         difference_hist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    #                         far = str(round(far,2))
    #                         difference_hist = str(round(difference_hist,2))
                            


    #                         # distance = round(distance,2)
    #                         # distance_tmp = str(round(distance_tmp,2))

    #                         # output the image and metadata
    #                         try:
    #                             Image.fromarray(low_image, 'RGB').save(low_resolution_path + '/' +str(out_number) + '.jpg')
    #                             Image.fromarray(high_image, 'RGB').save(high_resolution_path + '/' + str(out_number) + '.jpg')
    #                             Image.fromarray(out_low_image, 'RGB').save(low_resolution_big_path +'/'+str(out_number) + '.jpg')
    #                             Image.fromarray(out_high_image, 'RGB').save(high_resolution_big_path + '/'+ str(out_number) + '.jpg')

    #                             Image.fromarray(out_image, 'RGB').save(out_path + '/'+ str(out_number) + '.jpg')

    #                             far_list.append(far)
    #                             hist_list.append(difference_hist)
    #                             name_list.append(name)
    #                             point_name_list.append(point_name)
    #                             high_distance_list.append(distance)
    #                             low_distance_list.append(distance_tmp)
    #                             print(out_number , name ,point_name,len(far_list),len(name_list))
    #                             out_number += 1
    #                         except:
    #                             pass

    # with open(csvfile, "w") as output:
    #     writer = csv.writer(output, lineterminator='\n')
    #     for s in range(len(name_list)):
    #         writer.writerow([name_list[s],point_name_list[s],far_list[s],hist_list[s],high_distance_list[s],low_distance_list[s]])
                                
