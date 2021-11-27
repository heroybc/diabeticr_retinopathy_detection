''''
 * @Descripttion: train basecode
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-16 10:38:54
 '''
import os
import sys 
import nori2 as nori
from tqdm import tqdm
from loguru import logger
import json
import refile
import random
sys.path.append(".")
from utils.json_helper import save_json_items,load_json_items

def nori2imgname(annotation_nori_path):
    nori_imgname_dict = {}
    annots_nori = load_json_items(annotation_nori_path) 

    for annot_nori in annots_nori:
        imgname = annot_nori['fpath']
        nori_id = annot_nori['nori_id']
        imgname = os.path.basename(imgname)   # .jpeg
        imgname = imgname.split("-")[0].split(".")[0]
        nori_imgname_dict[imgname] = nori_id
    return nori_imgname_dict
    

def transcribe_to_sds(prompt, annotation_json_path, annotation_nori_path, output_sds_path, nori_imgname_dict):

    annots_json =  load_json_items(annotation_json_path)
    annots_nori = load_json_items(annotation_nori_path) 
    random.shuffle(annots_json)
    random.shuffle(annots_nori)

    sds = []
    error_list = []
    for annot_json in tqdm(annots_json):
        boxes = []

        boxes.append({
            'type': 'gt_box',
            'class_name':  annot_json['label_fine'],
            'x': 0,
            'y': 0,
            'w': 0,
            'h': 0,
            'extra': {
                'original':  annot_json['label_rough'],
            }
        })
        img_name = annot_json['img_name']
        if img_name in nori_imgname_dict:
            nori_id = nori_imgname_dict[str(img_name)]
        else :
            error_list.append(img_name)
            continue
        d = {
            'url': f'nori://{nori_id}',
            'image_width': annot_json['label_width'],
            'image_height': annot_json['label_height'],
            'boxes': boxes,
            'extra': annot_json,
        }
        sds.append(d)
    save_json_items(output_sds_path, sds)

    logger.info(f'sds: {output_sds_path}')


def main():
    nori_imgname_dict = nori2imgname('./megvii/kaggle_train.odgt')
    transcribe_to_sds(
        'train',
        './megvii/labels_traintest.json',
        './megvii/kaggle_train.odgt',
        './megvii/kaggle_train.sds',
        nori_imgname_dict
    )
    transcribe_to_sds(
        'val',
        './megvii/labels_traintest.json',
        './megvii/kaggle_val.odgt',
        './megvii/kaggle_val.sds',
        nori_imgname_dict
    )


if __name__ == '__main__':
    main()
