''''
 * @Descripttion: load datasets basecode
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-16 11:38:54
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
from torch.utils.data import BatchSampler,DistributedSampler,Dataset, DataLoader

from json_helper import save_json_items,load_json_items

def read_csv_label(csv_input_name = '../../data/sample/sampleSubmission.csv'):

    #csv_input_name = '../../data/sample/sampleSubmission.csv'
    #json_output_name = '../../data/sample/all_labels.json' 
    label_dict = {}
    label_json = []

    with open(csv_input_name, 'r') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            label_dict[line[0]] = line[1]
            
    return label_dict
    
class MyDataLoader(Dataset):
    def __init__(self, img_dir, label_dir, imgSize=(380, 380),  sampleNum=None, PreprocFun=None): 
        self.img_dir = img_dir
        self.label_dir = label_dir[0]
        self.img_paths = []
        self.sampleNum = sampleNum
        i_count = 0
        for i in range(len(img_dir)):
            image_paths_list =  paths.list_images(img_dir[i])
            print('image_paths_list done')
            for image_path in tqdm(image_paths_list):  
                    self.img_paths.append(image_path)  
        random.shuffle(self.img_paths)
        if not (self.sampleNum is None or  self.sampleNum > len(self.img_paths)):
            self.img_paths = self.img_paths[:self.sampleNum]
        self.img_size = imgSize

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self,):
            return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        #print(filename)
        '''
         # add aug and vis after aug img
        str_path = filename.split('/')
        if (str_path[3] != 'train') &  (str_path[3] != 'train-client')  &  (str_path[2] == 'train-ccpdgen-v40') :
            #Image = cv2.imread(filename)
            #cv2.imwrite('./1.jpg',Image)
            Image = self.add_aug(filename)
            #cv2.imwrite('./2.jpg',Image)
        else:
            Image = cv2.imread(filename)
        '''

        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        #imgname = imgname.split("-")[0].split("_")[0]
        #print(imgname)
        label = []
        label_dict = {}
        print('the path of label is {}'.format(self.label_dir))
        label_dict = load_json_items(self.label_dir)
        label.append(label_dict['label_fine'])
        #imgname = int(imgname)
        #label.append(int(imgname/10))    # 标注按照情况修改

        '''
        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"
        '''
            
        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True

    def r(self,val):
        return int(np.random.random() * val)

    def add_aug(self,filename):
        image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hlsImg[:, :, 2] = (1.0 + (100+self.r(50)) / float(100)) * hlsImg[:, :, 2]  # -100~+100  # 对比度
        hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
        lsImg = lsImg.astype(np.uint8)
        lsImg = cv2.blur(lsImg, (1+self.r(8), 1+self.r(8)))   # 高斯模糊
        return lsImg
        
def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)
        
if __name__ == "__main__":
    
    dataset = MyDataLoader(['../data/sample/sample'], ['../data/drd_data_all/labels_traintest.json'],(380, 380))   
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print('data length is {}'.format(len(dataset)))
    for imgs, labels, lengths in dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))      
        break
    
