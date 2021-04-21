''''
 * @Descripttion: divide train and val datset 
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-20 12:38:54
 '''
import torch
from imutils import paths
import numpy as np
import random
import cv2
import csv
import os
from tqdm import tqdm
import glob
import shutil
from torch.utils.data import BatchSampler,DistributedSampler,Dataset, DataLoader
from refile import smart_listdir, smart_isdir, smart_isfile, smart_open, s3_path_join

from json_helper import save_json_items,load_json_items

def trans_data(objects,img_outpath):  # 分为训练集和测试集,产生data文件

    for a in tqdm(objects):
        id_img = str(a['img_id'])
        data_type = str(a['tag'])
        img_path = str(a['fpath'])

        imgname = os.path.basename(img_path)   # .jpeg
        #print(imgname)
        #img = cv2.imread(img_path)

        if data_type == 'test' :
            outpath = os.path.join(img_outpath+'/val', imgname) # 输出照片路径拼接
            shutil.copy(img_path, outpath)
            
        elif data_type == 'train' :
            outpath = os.path.join(img_outpath+'/train',imgname) # 输出照片路径拼接
            shutil.copy(img_path, outpath)

        else:
            print('ERROR: unknown but not written in the label file !')
            exit()

if __name__ == "__main__":

    img_outpath = '../data/'
    json_path = '../data/drd_data_all/labels_traintest.json'
    objects = load_json_items(json_path)
    trans_data(objects,img_outpath)