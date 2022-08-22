import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AdamW
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

#from transformers import BertPreTrainedModel


from transformers import BertModel

from loader import load_train
from loader import load_dev

from loader import map_id_rel
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)



USE_CUDA = torch.cuda.is_available()
#USE_CUDA=False

#加载训练数据
data=load_train()
train_text=data['text']
train_mask=data['mask']
train_label=data['label']

train_text = [t.numpy() for t in train_text]
train_mask = [t.numpy() for t in train_mask]

train_text=torch.tensor(train_text)
train_mask=torch.tensor(train_mask)
train_label=torch.tensor(train_label)


#加载测试数据
data=load_dev()
dev_text=data['text']
dev_mask=data['mask']
dev_label=data['label']

dev_text = [ t.numpy() for t in dev_text]
dev_mask = [ t.numpy() for t in dev_mask]

dev_text=torch.tensor(dev_text)
dev_mask=torch.tensor(dev_mask)
dev_label=torch.tensor(dev_label)

print("--train data--")
print(train_text.shape)
print(train_mask.shape)
print(train_label.shape)

print("--eval data--")
print(dev_text.shape)
print(dev_mask.shape)
print(dev_label.shape)

# exit()
#USE_CUDA=False

if USE_CUDA:
    print("using GPU")
else:
    print('did not use GPU')

train_dataset = torch.utils.data.TensorDataset(train_text,train_mask,train_label)
dev_dataset = torch.utils.data.TensorDataset(dev_text,dev_mask,dev_label)

def get_model():
    labels_num=len(rel2id)
    from model import BERT_Classifier
    model = BERT_Classifier(labels_num)
    return model


def eval(net,dataset, batch_size):
    net.eval()#训练完train_datasets之后，model要来测试样本了。在model(test_datasets)之前，????????????????????????
    # 需要加上model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
    #eval（）时，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)#数据读取  解决训练数据不能被batchsize整除：drop_last=True
    # 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，
    # 后续只需要再包装成Variable即可作为模型的输入，因此该接口有点承上启下的作用，比较重要。
    with torch.no_grad():#torch.no _ grad() 一般用于神经网络的推理阶段, 表示张量的计算过程中无需计算梯度
        correct = 0
        total=0
        iter = 0
        for text,mask, y in train_iter:
            iter += 1
            if text.size(0)!=batch_size:
                #numpy.size(a, axis=None)
                # a：输入的矩阵
                # axis：int型的可选参数，指定返回哪一维的元素个数。当没有指定时，返回整个矩阵的元素个数。
                break
            text=text.reshape(batch_size,-1)##改变维度为batch_size行、d列 （-1表示列数自动计算，d= a*b /m ）
            mask = mask.reshape(batch_size, -1)
            
            if USE_CUDA:
                text=text.cuda()
                mask=mask.cuda()
                y=y.cuda()

            outputs= net(text, mask,y)#????????
            #print(y)
            loss, logits = outputs[0],outputs[1]
            _, predicted = torch.max(logits.data, 1)#每行（”1的含义“）的最大值
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
            # torch.eq()方法详解
            # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
            # correct += predicted.data.eq(y.data).gpu().sum()  ####GPU????
            s = ("Acc:%.3f" %((1.0*correct.numpy())/total))
        acc= (1.0*correct.numpy())/total
        print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total, "Acc:", acc)
        return acc


def train(net,dataset,num_epochs, learning_rate,  batch_size):
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0)#SGD：随机梯度下降算法
    #optimizer = AdamW(net.parameters(), lr=learning_rate)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    pre=0
    losses=[]
    epochAcc=[]
    devAcc=[]
    epochNums=[]
    epochCount=0
    for epoch in range(num_epochs):
        correct = 0
        total=0
        iter = 0
        for text,mask, y in train_iter:
            iter += 1
            optimizer.zero_grad()#111111先将梯度归零（optimizer.zero_grad()），进来一个batch的数据，计算一次梯度，更新一次网络
            #print(type(y))
            #print(y)
            if text.size(0)!=batch_size:
                break
            text=text.reshape(batch_size,-1)
            mask = mask.reshape(batch_size, -1)
            if USE_CUDA:
                text=text.cuda()
                mask=mask.cuda()
                y = y.cuda()
            #print(text.shape)
            loss, logits= net(text, mask,y)

            loss.backward()#22222然后反向传播计算得到每个参数的梯度值（loss.backward()），
            optimizer.step()#33333最后通过梯度下降执行一步参数更新（optimizer.step()）
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
        loss = loss.detach().cpu()
        epochTrainacc=correct.cpu().numpy().tolist()/total
        print("epoch ", str(epoch)," loss: ", loss.mean().numpy().tolist(),"right", correct.cpu().numpy().tolist(), "total", total, "Acc:", epochTrainacc)
        acc = eval(model, dev_dataset, 32)
        if acc > pre:
            pre = acc
            torch.save(model, str(acc)+'.pth')
        epochCount+=1
        losses.append(loss)
        epochAcc.append(epochTrainacc)
        devAcc.append(acc)
        epochNums.append(epochCount)
    return losses,epochAcc,devAcc,epochNums


model=get_model()
#model=nn.DataParallel(model,device_ids=[0,1])
if USE_CUDA:
    model=model.cuda()

# 绘制折线图
import matplotlib as mpl
import matplotlib.pyplot as plt
def drawResult(losses,epochAcc,devAcc,epochNums):
    # # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    # plt.plot(epochNums, losses, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='losses')
    # # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    # plt.legend(loc="upper right")
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    #
    # plt.show()
    print('losses：{0} lossestype：{1}'.format(losses, type(losses)))
    print('epochAcc：{0} epochAcc：{1}'.format(epochAcc, type(epochAcc)))
    print('devAcc：{0} devAcc：{1}'.format(devAcc, type(devAcc)))
    print('epochNums：{0} epochNums：{1}'.format(epochNums, type(epochNums)))
    plt.subplot(211)
    plt.plot(epochNums, losses, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='losses')
    # for x, y in zip(epochNums, losses):
    #     plt1.text(x, y, '%.0f' % y, fontdict={'fontsize': 14})
    # 画第2个图：散点图
    plt.xticks(epochNums)
    plt.legend()
    plt.subplot(212)
    plt.plot(epochNums, epochAcc, 'o-', color='#4169E1', alpha=0.8, linewidth=1, label='epochAcc')
    # plt.subplot(313)
    plt.plot(epochNums, devAcc, '*-', color='#4169E1', alpha=0.8, linewidth=1, label='devAcc')
    # for x, y in zip(epochNums, devAcc):
    #     plt2.text(x, y, '%.0f' % y, fontdict={'fontsize': 14})
    plt.xticks(epochNums)
    plt.legend()
    plt.show()

    # plt.savefig('demo.jpg')  # 保存该图片
    return

#eval(model,dev_dataset,8)

# train(model,train_dataset,10,0.002,4)
losses,epochAcc,devAcc,epochNums=train(model,train_dataset,10,0.002,4)
#eval(model,dev_dataset,8)
drawResult(losses,epochAcc,devAcc,epochNums)

#
#
# 画第1个图：折线图
#
# plt.subplot(211)
# plt.plot(epochNums, losses, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='losses')
# # 画第2个图：散点图
# plt.subplot(212)
#plt.plot(epochNums, epochAcc, 'o', color='#4169E1', alpha=0.8, linewidth=1, label='epochAcc')
#plt.plot(epochNums, devAcc, '*', color='#4169E1', alpha=0.8, linewidth=1, label='devAcc')
# plt.show()

#
#