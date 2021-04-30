# DRD-Net (diabetic-retinopathy-detection network)

### 此项目是@yangbinchao @zhaokaifei1 糖尿病视网膜病变检测

## 0. data
数据集下载：[点击跳转](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
数据集下载脚本：[点击跳转](https://github.com/gregwchase/eyenet/blob/master/src/download_data.sh)
kaggle数据集：
>训练数据集：24589 （0.7）
>验证数据集：10537（0.3）
训练集和测试集标注文件：../data/drd_data_all/labels_traintest.json

标注：
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR

## 1. train
`python3 train.py`
    

## 2. 特征可视化
`python3 grad_cam_vis.py`


## 参考
https://github.com/gregwchase/eyenet


## log
> * 2021-4-29  eval
> * 2021-4-25  train调试
> * 2021-4-21  修改标注文件更加全面，产生NORI
> * 2021-4-20 test vscode git 5
> * 2021-4-19 更新load data load csv
> * 2021-4-16 更新basecode
> * 2021-4-15 得到rawcode

 ## megvii
 s3://yangbinchao/drd/kaggle_val.nori
 s3://yangbinchao/drd/kaggle_train.nori
 https://oss.iap.wh-a.brainpp.cn/snapdet-tmp/data-visualize/yangbinchao/2021-04-21-20-09-36.3a335076-bf7a-4df8-abd2-54a3f7a56661/#/0


