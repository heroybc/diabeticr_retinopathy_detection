''''
 * @Descripttion: csv to json code
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-18 12:38:54
 '''

 #!/usr/bin/env python
# coding=utf-8
import os
import cv2
import json
import csv
from tqdm import tqdm
import codecs
import random

from json_helper import save_json_items,load_json_items

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def read_csv_label(csv_input_name):

    img_path = '../data/drd_data_all/images'
    
    obj_structure = {}
    objects = []

    with open(csv_input_name, 'r') as myFile:
        lines=csv.reader(myFile)
        lines  = list(lines)
        random.shuffle(lines)
        img_id = 0
        for line in tqdm(lines):
            img_id += 1
            img_name = line[0]
            img_dir = os.path.join(img_path,img_name+'.jpeg')
            
            img = cv2.imread(img_dir)
            img_h = img.shape[0]   #（图片高度）
            img_w = img.shape[1]
            if line[1] == '0':
                label_rough = 'positive'
            else:
                label_rough = 'negative'
            if img_id < len(lines)*0.3 :
                tag = 'test'
            else:
                tag = 'train'
            label_fine = line[1]
            obj_structure = {}
            obj_structure['img_name'] = img_name
            obj_structure['label_rough'] = label_rough
            obj_structure['label_fine'] = label_fine
            obj_structure['img_id'] = img_id
            obj_structure['label_width'] = img_w
            obj_structure['label_height'] = img_h
            obj_structure['fpath'] = img_dir
            obj_structure['tag'] = tag
            objects.append(obj_structure)
    return objects


if __name__ == "__main__":

    json_output_name = '../data/drd_data_all/labels_traintest.json' 
    csv_input_name = '../data/drd_data_all/labels.csv'
    objects = read_csv_label(csv_input_name)
    save_json_items(json_output_name,objects)