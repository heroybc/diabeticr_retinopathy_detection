''''
 * @Descripttion: train basecode
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-16 10:38:54
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
from loss.focal_loss import FocalLoss

'''
efficientnet-b0- 224 1280
efficientnet-b1- 240 1280
efficientnet-b2- 260 1408
efficientnet-b3- 300 1536
efficientnet-b4- 380 1792
efficientnet-b5- 456 2048
efficientnet-b6- 528 2304
efficientnet-b7- 600 2560
'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='./weights/efficientnet-b4.pth',help='Input pre-train model weights path') # efficientnet-b7-dcc49843 efficientnet-b4-6ed6700e efficientnet-b1-f1951068
    parser.add_argument('--img_size',  default=(600,600),help='the input image size')
    parser.add_argument('--img_dirs_train', type=str, default='../data/drd_data_all/train',help='Input train datasets path')  # ../data/train
    parser.add_argument('--img_dirs_val', type=str, default='../data/drd_data_all/val', help='Input val datasets path')
    parser.add_argument('--label_train_path', type=str, default='../data/drd_data_all/labels_train.json', help='Input label datasets path')
    parser.add_argument('--label_val_path', type=str, default='../data/drd_data_all/labels_val.json', help='Input label datasets path')
    parser.add_argument('--batch_size', default= 32 ,help='the batch size')
    parser.add_argument('--lr', default= 0.001 ,help='the lr')
    parser.add_argument('--epoch', default= 15 , help='the number of epoch')
    parser.add_argument('--save_modeldir', default="./checkpoint", help='the save model path')
    parser.add_argument('--save_rundir', default="./runs", help='the save tensorboard runs path')


    args = parser.parse_args()
    return args

def train(args, model, dataloader, device, criterion, optimizer, scheduler):
    start_time = time.time()
    total_iters = 0
    print('training kicked off..')
    print('-' * 10) 
    best_acc = 0.0
    writer = SummaryWriter(os.path.join(args.save_rundir , str(time.strftime("%m-%d-%H-%M-%S", time.localtime()))+'-B'+str(args.batch_size) +'-E'+str(args.epoch))) 
    save_modeldir = os.path.join(args.save_modeldir , str(time.strftime("%m-%d-%H-%M-%S", time.localtime()))+'-B'+ str(args.batch_size) +'-E'+ str(args.epoch))
    fake_img = torch.randn(1, 3, 380, 380) #生成假的图片作为输入
    #writer.add_graph(model, fake_img)

    for epoch in range(args.epoch): 
        
        train_running_loss = 0.0
        train_final_loss = 0.0
        test_running_loss = 0.0
        accuracy = 0.0

        for name, param in model.named_parameters():
            writer.add_histogram(name , param.clone().cpu().data.numpy(), epoch)
        
        model.train()
        since = time.time()
        
        if os.path.exists(save_modeldir):
            pass
        else:
            os.makedirs(save_modeldir)

        for imgs, labels, lengths in dataloader['train']:
            labels = labels.long()
            #print(labels)
            imgs, labels = imgs.to(device), labels.to(device)  # inputs
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(imgs)
                train_loss = criterion(outputs, labels)
                train_loss.backward()
                optimizer.step()
                train_running_loss += train_loss.item()
                total_iters += 1
                
                #x = vutils.make_grid(imgs, normalize=True, scale_each=True)  # 可视化输入图片
                #writer.add_image('input image', x, total_iters)  
            
                if total_iters % 10 == 0:    
                    correct = 0
                    total = 0
                    y_pred = []
                    y_true = []
                    pred_label = outputs.cpu().detach().numpy()
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred.append(predicted)            
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    TP = 0
                    start = 0
                    total = lengths
                    for i, length in enumerate(lengths):
                        label = labels[start:start+length]
                        start += length
                        if np.array_equal(np.array(pred_label), label.cpu().numpy()):
                            TP += 1
                        
                    time_elapsed = (time.time() - since) / 100
                    since = time.time()

                    writer.add_scalar('train_criterion_loss', train_running_loss / 10, total_iters)   # 损失
                    writer.add_scalar('train_accuracy', accuracy , total_iters)   # 精度

                    for p in  optimizer.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                            .format(epoch, args.epoch-1, total_iters, train_running_loss / 10,  accuracy, time_elapsed, lr))
                    train_running_loss = 0.0

                if total_iters % 380 == 0:    
                    PATH = save_modeldir + '/' +  str(total_iters) +'_'+ str(epoch)+ '.pth'
                    torch.save(model.state_dict(), PATH)
                    model.eval()
                    correct = 0
                    total = 0
                    y_pred = []
                    y_true = []
                    model.load_state_dict(torch.load(PATH))
                    with torch.no_grad():
                        for imgs, labels, lengths in tqdm(dataloader['val']):
                            imgs, labels = imgs.to(device), labels.to(device)  # inputs
                            outputs = model(imgs)
                            _, predicted = torch.max(outputs.data, 1)
                            y_pred.append(predicted)
                            y_true.append(labels)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        acc = correct / total
                    writer.add_scalar('val_accuracy', acc , total_iters)   # 精度

                    if best_acc <= acc:
                        #print('交换之前最佳精度：{}'.format(best_acc))
                        best_acc = acc
                        best_iters = total_iters
                    print('Current Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters))
                    print("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, acc))
                    print('the model dir is {}'.format(save_modeldir))
                    print('-' * 10) 
        scheduler.step()   

                    

def main():
    args = get_args()
    print(args)

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(not(torch.cuda.is_available())):  
        print('\ncuda is not available !\n')
    print("device: {}".format(device))  
    
    model = EfficientNet.from_pretrained('efficientnet-b4',weights_path=args.weights_path, num_classes=5).cuda()
    model = torch.nn.DataParallel(model).cuda()  # Multiple gpu
    print("model loaded")
    #writer.add_graph(model, t.Tensor(args.batch_size))

    dataset = {'train': MyDataLoader([args.img_dirs_train], [args.label_train_path], args.img_size),  # 加载数据集strftime
                          'val': MyDataLoader([args.img_dirs_val], [args.label_val_path], args.img_size, rand_size=False)}
    print("MyDataLoader loaded")
    dataloader = {'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn),
                'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)}
    print("dataloader loaded")
    print('training dataset loaded with length : {}'.format(len(dataset['train'])))
    print('validation dataset loaded with length : {}'.format(len(dataset['val'])))

    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(5, alpha= torch.Tensor([1.4,14,7,30,30]))  #  , alpha= torch.Tensor([1.4,14,7,30,30])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99)) #optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train(args, model, dataloader, device, criterion, optimizer, scheduler)  # check your input paras is true or flase 

if __name__ == '__main__':
    main()
