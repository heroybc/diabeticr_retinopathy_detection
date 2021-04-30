实验1 复现原始代码并训练
数据: 全部kaggle数据直接resize为380*380
参数: bs=64,epoch=8,gpu=8,model=b4,lr=0.001,dir=04-22-20-46-23-B32-E8
结果: Current Best Accuracy: 0.8103 in iters: 5700
分析: 进度跳跃明显，怀疑是样本间差异大导致

实验2 efficientnet-b0训练测试
数据: 全部kaggle数据裁剪为224*224的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b0,lr=0.001,dir=04-26-10-37-55-B64-E15
结果: Current Best Accuracy: 0.7644 in iters: 3420

实验3 efficientnet-b1训练测试
数据: 全部kaggle数据裁剪为240*240的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b1,lr=0.001,dir=04-26-10-40-01-B64-E15
结果: Current Best Accuracy: 0.7712 in iters: 3040

实验4 efficientnet-b2训练测试
数据: 全部kaggle数据裁剪为260*260的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b2,lr=0.001,dir=04-26-14-50-34-B64-E15
结果: Current Best Accuracy: 0.7714 in iters: 3040

实验4 efficientnet-b3训练测试
数据: 全部kaggle数据裁剪为300*300的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b3,lr=0.001,dir=04-26-14-51-33-B64-E15
结果: Current Best Accuracy: 0.7806 in iters: 3420

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-23-18-04-43-B64-E15
结果: Current Best Accuracy: 0.7892 in iters: 3040

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-27-10-49-20-B64-E15
设置: focal loss
结果: Current Best Accuracy: 0.7880 in iters: 4180

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-27-11-26-03-B64-E15
设置: focal loss,  data load shuffle=True
结果: Current Best Accuracy: 0.8054 in iters: 5700

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b4,lr=0.0001,dir=04-27-13-36-05-B64-E15
设置: focal loss,  data load shuffle=True，修改学习率为0.0001
结果: Current Best Accuracy: 0.7428 in iters: 5700

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-27-11-33-02-B64-E15
设置: FocalLoss(5, alpha=[0.25,0.75,0.75,0.75,0.75]),  data load shuffle=True
结果: Current Best Accuracy: 0.7743 in iters: 5320

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-27-13-33-35-B64-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True
结果: Current Best Accuracy: 0.8214 in iters: 2280

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=128,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-27-16-09-02-B128-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True，扩大bs
结果: Current Best Accuracy: 0.8137 in iters: 1900

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=32,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-27-18-21-27-B32-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True，缩小bs
结果: Current Best Accuracy: 0.8317 in iters: 4940
分析: 在Iters: 010700时出现梯度爆炸，减小epoch为10以下

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为随机6种大小的正方形，每个bs都一致的大小（与每个batch第一张的大小一致）
参数: bs=32,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-28-15-32-31-B32-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True
结果: 梯度爆炸

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为随机6种大小的正方形，每个bs都一致的大小（与每个batch第一张的大小一致）
参数: bs=32,epoch=15,gpu=8,model=b4,lr=0.0001,dir=04-28-16-04-07-B32-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True，降低学习率
结果: Current Best Accuracy: 0.7414 in iters: 10640
分析:  Loss不下降，精度不上

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为600*600大小的正方形
参数: bs=32,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-28-21-05-14-B32-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True
结果: Current Best Accuracy: 0.8443 in iters: 3800

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为600*600大小的正方形
参数: bs=32,epoch=15,gpu=8,model=b4,lr=0.0001,dir=04-28-21-02-49-B32-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True，降低学习率
结果: Current Best Accuracy: 0.8343 in iters: 11400

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为600*600大小的正方形
参数: bs=32,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-29-10-22-15-B32-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True，optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
结果: Current Best Accuracy: 0.8365 in iters: 5700

实验2 efficientnet-b4训练测试
数据: 全部kaggle数据裁剪为600*600大小的正方形
参数: bs=32,epoch=15,gpu=8,model=b4,lr=0.001,dir=04-29-11-05-07-B32-E15
设置: FocalLoss(5, alpha=[1.4,14,7,30,30]),  data load shuffle=True，optim.Adam，每个epoch让lr*0.9
结果: Current Best Accuracy: 0.8416 in iters: 4560

实验4 efficientnet-b5训练测试
数据: 全部kaggle数据裁剪为456*456的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b5,lr=0.001,dir=04-26-16-25-45-B64-E15
结果: Current Best Accuracy: 0.7973 in iters: 3040

实验4 efficientnet-b5训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b5,lr=0.001,dir=04-23-19-58-59-B64-E15
结果: Current Best Accuracy: 0.7755 in iters: 2660

实验4 efficientnet-b6训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=64,epoch=15,gpu=8,model=b5,lr=0.001,dir=04-24-20-28-50-B64-E15
结果: Current Best Accuracy: 0.7578 in iters: 2280

实验4 efficientnet-b6训练测试
数据: 全部kaggle数据裁剪为528*528的正方形，眼球居中与样本高同宽
参数: bs=32,epoch=15,gpu=8,model=b6,lr=0.001,dir=04-26-16-44-18-B32-E15
结果: Current Best Accuracy: 0.8226 in iters: 9120

实验4 efficientnet-b7训练测试
数据: 全部kaggle数据裁剪为380*380的正方形，眼球居中与样本高同宽
参数: bs=32,epoch=15,gpu=8,model=b7,lr=0.001,dir=04-24-20-32-43-B32-E15
结果: Current Best Accuracy: 0.8042 in iters: 9120

实验4 efficientnet-b7训练测试
数据: 全部kaggle数据裁剪为600*600的正方形，眼球居中与样本高同宽
参数: bs=16,epoch=15,gpu=8,model=b7,lr=0.001,dir=04-26-17-35-27-B16-E15
设置: 损失函数设置为focal loss
结果: Current Best Accuracy: 0.8325 in iters: 19760

实验4 efficientnet-b7训练测试
数据: 全部kaggle数据裁剪为600*600的正方形，眼球居中与样本高同宽
参数: bs=16,epoch=15,gpu=8,model=b7,lr=0.001,dir=04-26-18-46-15-B16-E15
结果: Current Best Accuracy: 0.8383 in iters: 4560
分析: 训练后期出现梯度爆炸，可能原因是学习率过大