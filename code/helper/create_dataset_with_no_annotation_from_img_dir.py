"""
本脚本用于将一个文件夹的图片，上传到nori，并生成type2格式的无标注数据集。并自动完成nori加速、数据集可视化的功能
sample cmd：
    python3 create_dataset_with_no_annotation_from_img_dir.py --img_dir_path /data/datapp_commercialize/01_zhongyizhengqi_mingchuliangzao/00_data/data_from_custom/20200707/images --nori_dir_path s3://data-wjx/data/datapp_commercialize/01_zhongyizhengqi_mingchuliangzao/00_data/data_from_custom/20200707/images.nori --res_dataset_path s3://data-wjx/data/datapp_commercialize/01_zhongyizhengqi_mingchuliangzao/00_data/data_from_custom/20200707/images.json
"""


import os
import sys
import cv2
import json
import argparse
from tqdm import tqdm

import nori2 as nori
from refile import s3_exists, smart_open

sys.path.append("/data/mypro/")
from file_helper import gather_files


def parse_arg():

    parser = argparse.ArgumentParser(description="输出图片文件夹，将图片打成nori并生成一个没有标注的type2格式的数据集")
    parser.add_argument("--img_dir_path", dest="img_dir_path",
                        required=True, type=str, help="path to img dir")
    parser.add_argument("--nori_dir_path", dest="nori_dir_path",
                        required=True, type=str, help="path to nori dir to save images, s3 path is ok")
    parser.add_argument("--res_dataset_path", dest="res_dataset_path",
                        required=True, type=str, help="path to result dataset, s3 path is ok")
    parser.add_argument("--img_extension_list", default=["jpeg", "png","jpg"], nargs='+', type=str, help="extension list of imgs")

    args = parser.parse_args()

    print()
    print(args)
    print()

    return args


def check_args(args):
    assert os.path.exists(args.img_dir_path)

    if args.nori_dir_path.startswith("s3://"):
        assert not s3_exists(args.nori_dir_path), args.nori_dir_path
    else:
        assert not os.path.exists(args.nori_dir_path), args.nori_dir_path

    if args.res_dataset_path.startswith("s3://"):
        assert not s3_exists(args.res_dataset_path), args.res_dataset_path
    else:
        assert not os.path.exists(args.res_dataset_path), args.res_dataset_path


def save_imgs_to_nori(img_dir_path, nori_dir_path, img_extension_list):
    img_paths = gather_files(img_dir_path, keep_suffixs=img_extension_list)
    res_items = []
    with nori.open(nori_dir_path, "w") as nw:
        for img_path in tqdm(img_paths):
            img = cv2.imread(img_path)
            if img is None:
                print("img is None: {}".format(img_path))
                continue
            img_shape = img.shape
            ret, img = cv2.imencode(".jpeg", img)
            if ret:
                nori_id = nw.put(img.tostring(), filename=img_path)
            else:
                print("cv2.imencode error !!!")
                exit()
            res_items.append({
                "img_path": img_path,
                "img_shape": img_shape,
                "nori_id": nori_id,
                "nori_path": nori_dir_path,
            })
            # break
    return res_items


def create_type2_dataset(res_dataset_path, dataset_items):
    res_items = []
    for dataset_info in dataset_items:
        nori_id = dataset_info["nori_id"]
        nori_path = dataset_info["nori_path"]
        img_path = dataset_info["img_path"]
        image_height = dataset_info["img_shape"][0]
        image_width = dataset_info["img_shape"][1]
        item = {
            "nori_id": nori_id,
            "nori_path": nori_path,
            "fpath": img_path,
            "image_height": image_height,
            "image_width": image_width,
            "gtboxes": [],
        }
        res_items.append(item)

    with smart_open(res_dataset_path, "w") as f:
        f.write("\n".join([json.dumps(x) for x in res_items]))
    return len(res_items)


def speedup_nori(nori_dir_path):
    cmd = "nori speedup {} --on --replica 2".format(nori_dir_path)
    print("cmd = {}".format(cmd))
    res_code = os.system(cmd)
    if res_code != 0:
        print("nori 加速失败！！！")


def vis_dataset(res_dataset_path, len_items=500):
    cmd = "hubble det vis -s {} -l {} {}".format("fr", len_items, res_dataset_path)
    print("cmd = {}".format(cmd))
    res_code = os.system(cmd)
    if res_code != 0:
        print("vis dataset失败！！！")


def handle(args):
    # step 0
    check_args(args)

    # step 1
    print("save_imgs_to_nori...")
    dataset_items = save_imgs_to_nori(args.img_dir_path, args.nori_dir_path, args.img_extension_list)

    # step 2
    print("create_type2_dataset...")
    len_items = create_type2_dataset(args.res_dataset_path, dataset_items)

    # step 3
    print("speedup_nori")
    speedup_nori(args.nori_dir_path)

    # step 4
    print("vis dataset")
    vis_dataset(args.res_dataset_path, len_items)


def main():
    args = parse_arg()

    handle(args)

    print()
    print("done")
    print("nori_dir_path = {}".format(args.nori_dir_path))
    print("res_dataset_path = {}".format(args.res_dataset_path))
    print()


if __name__ == "__main__":
    main()
