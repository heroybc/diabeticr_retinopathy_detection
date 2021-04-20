''''
 * @Descripttion: train basecode
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-16 10:38:54
 '''

import os
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

from utils.load_data import MyDataLoader,collate_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='./weights/efficientnet-b4-6ed6700e.pth',help='Input pre-train model weights path')
    parser.add_argument('--img_size',  default=(380,380),help='the input image size')
    parser.add_argument('--img_dirs_train', type=str, default='../data/sample/sample',help='Input train datasets path')  # ../data/train
    parser.add_argument('--img_dirs_val', type=str, default='../data/sample/sample', help='Input val datasets path')
    parser.add_argument('--label_path', type=str, default='../data/sample/sampleSubmission.csv', help='Input label datasets path')
    parser.add_argument('--batch_size', default= 128 ,help='the batch size')
    parser.add_argument('--epoch', default= 8 , help='the number of epoch')
    parser.add_argument('--save_modeldir', default="./checkpoint", help='the save model path')


    args = parser.parse_args()
    return args

def train(args, model, dataloader, device, criterion, optimizer):
    start_time = time.time()
    total_iters = 0
    print('training kicked off..')
    print('-' * 10) 
    
    for epoch in range(args.epoch): 
        
        train_running_loss = 0.0
        train_final_loss = 0.0
        test_running_loss = 0.0
        best_acc = 0.0

        model.train()
        since = time.time()
        save_modeldir = os.path.join(args.save_modeldir , str(time.strftime("%m-%d-%H-%M-%S", time.localtime())))
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
            
                if total_iters % 2 == 0:    
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

                    train_running_loss = 0.0
                    time_elapsed = (time.time() - since) / 100
                    since = time.time()

                    for p in  optimizer.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                            .format(epoch, args.epoch-1, total_iters, train_running_loss / 2,  accuracy, time_elapsed, lr))
                    PATH = save_modeldir + '/' +  str(epoch) + '_' + str(accuracy) + '.pth'
                    torch.save(model.state_dict(), PATH)
                    

                if total_iters % 5 == 0:    
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
                    accuracy = correct / total

                    if best_acc <= accuracy:
                        best_acc = accuracy
                        best_iters = total_iters
                    print('Current Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters))
                    print("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, accuracy))

                    PATH = save_modeldir + '/' +  str(epoch) + '_' + str(accuracy) + '.pth'
                    torch.save(model.state_dict(), PATH)

def main():
    args = get_args()

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(not(torch.cuda.is_available())):  
        print('\ncuda is not available !\n')
    print("device: {}".format(device))  
    
    model = EfficientNet.from_pretrained('efficientnet-b4',weights_path=args.weights_path, num_classes=5).cuda()
    model = torch.nn.DataParallel(model).cuda()  # Multiple gpu
    print("model loaded")

    dataset = {'train': MyDataLoader([args.img_dirs_train], [args.label_path], args.img_size),  # 加载数据集strftime
                          'val': MyDataLoader([args.img_dirs_val], [args.label_path], args.img_size)}
    print("MyDataLoader loaded")
    dataloader = {'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn),
                'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn)}
    print("dataloader loaded")
    print('training dataset loaded with length : {}'.format(len(dataset['train'])))
    print('validation dataset loaded with length : {}'.format(len(dataset['val'])))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(args, model, dataloader, device, criterion, optimizer)  # check your input paras is true or flase 

if __name__ == '__main__':
    main()
