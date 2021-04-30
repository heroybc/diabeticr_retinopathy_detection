''''
 * @Descripttion: eval basecode
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-26 10:38:54
 '''

import os
import cv2
import time
import torch
import torchvision
import argparse
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage import io, transform
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torchvision.utils as vutils

from utils.load_data import MyDataLoader,collate_fn
from utils.fusion_matrix import confusion_matrix, plot_confusion_matrix

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/04-29-11-05-07-B32-E15/4560_5.pth',help='Input pre-train model weights path') #
    parser.add_argument('--weights_path', type=str, default='./weights/efficientnet-b4.pth',help='Input pre-train model weights path') # efficientnet-b7-dcc49843 efficientnet-b4-6ed6700e efficientnet-b1-f1951068
    parser.add_argument('--img_size',  default=(600,600),help='the input image size')
    parser.add_argument('--class_number',  default= 5 ,help='the class number size')
    parser.add_argument('--img_eval_path', type=str, default='../data/drd_data_all/val', help='Input eval datasets path')
    parser.add_argument('--label_eval_path', type=str, default='../data/drd_data_all/labels_val.json', help='Input label datasets path')
    parser.add_argument('--batch_size', default= 16 ,help='the batch size')
    args = parser.parse_args()
    return args

def eval(args, model, dataset, dataloader, device):
    PATH = args.checkpoint_path

    model.eval()
    correct = 0
    total = 0
    y_pred = None
    y_true = None

    model.load_state_dict(torch.load(PATH))

    for imgs, labels, lengths in tqdm(dataloader['eval']):
        imgs, labels = imgs.to(device), labels.to(device)  # inputs
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        if y_pred is None:
            y_pred = predicted
            y_true = labels
        else:
            y_pred = torch.cat((y_pred, predicted))
            y_true = torch.cat((y_true, labels))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    print('Eval Accuracy: {:.4f} and all iters is: {}'.format(acc, total))
    print('the model dir is {}'.format(PATH))
    print('-' * 10) 
    #print(y_pred,y_true)
    return y_pred,y_true

def fusion_matrix(args, y_pred,y_true):
    conf_matrix  = [[0 for i in range(args.class_number)]for j in range(args.class_number)]#torch.zeros(args.class_number, args.class_number)
    conf_matrix = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    conf_matrix = confusion_matrix(y_pred, y_true, conf_matrix=conf_matrix)
    plot_confusion_matrix(conf_matrix, classes=[0,1,2,3,4], normalize=False, title='Normalized confusion matrix')

def main():
    args = get_args()
    print(args)

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(not(torch.cuda.is_available())):  
        print('\ncuda is not available !\n')
    print("device: {}".format(device))  
    
    model = EfficientNet.from_pretrained('efficientnet-b4',weights_path=args.weights_path, num_classes=5).cuda()
    model = torch.nn.DataParallel(model).cuda()  # Multiple gpu
    dataset = {'eval': MyDataLoader([args.img_eval_path], [args.label_eval_path], args.img_size)}
    dataloader = {'eval': DataLoader(dataset['eval'], batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)}
    print("MyDataLoader \ model \ dataloader loaded")
    print('validation dataset loaded with length : {}'.format(len(dataset['eval'])))

    y_pred,y_true = eval(args, model, dataset, dataloader, device)  
    fusion_matrix(args, y_pred,y_true)

if __name__ == '__main__':
    main()
