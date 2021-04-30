''''
 * @Descripttion: 混淆矩阵相关代码
 * @version: v1.0.0
 * @Author: yangbinchao
 * @Date: 2021-04-29 10:38:54
*  @Use:
    # conf_matrix 需要是numpy格式
    # attack_types 是分类实验的类别，eg：attack_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    conf_matrix = torch.zeros(cfg.NUM_CLASSES, cfg.NUM_CLASSES)
    conf_matrix = analytics.confusion_matrix(prediction, labels=batch_labels, conf_matrix=conf_matrix)
    plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=False, title='Normalized confusion matrix')
 '''
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    #print(len(preds),len(labels))
    for i in range(len(preds)):#, labels:
        p=preds[i]
        t=labels[i]
        conf_matrix[int(p), int(t)] += 1
    return conf_matrix

'''
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
Input
- cm : 计算出的混淆矩阵的值
- classes : 混淆矩阵中每一行每一列对应的列
- normalize : True:显示百分比, False:显示个数
'''
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    exit()
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    print('123')
	# x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    print('456')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    print('789')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./demo/fusion_matrix.png")
    #plt.show()
