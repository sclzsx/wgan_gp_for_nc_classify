import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from vgg import vgg16
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from collections import Counter

from uc_dataset import UCDataset, class_names, class_ids, class_name2id, class_id2name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择用cpu还是gpu

batch_size = 1
max_epoch = 50

trainset_id = 1

model = vgg16().to(device)

if trainset_id == 0:    
    save_dir = 'results/aug/'
else:
    save_dir = 'results/noaug/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_dataset = UCDataset('../UCMerced_LandUse/train', image_size=256)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

test_dataset = UCDataset('../UCMerced_LandUse/test', image_size=256)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train():
    max_f1 = -1
    iter_per_epoch = len(train_dataloader)
    print_iter_freq = 0.25
    print_iter = int(iter_per_epoch) * print_iter_freq
    for epoch in range(1, max_epoch + 1):  # range的区间是左闭右开,所以加1
        model.train()  # 训练模式
        mean_loss = 0
        for i, (batch_data, batch_labels) in enumerate(train_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
            # print(batch_data.shape, batch_labels.shape)
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
            batch_labels = batch_labels.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式

            optimizer.zero_grad()  # 清零已有梯度

            batch_pred = model(batch_data)  # 前向传播,获得网络的输出
            batch_loss = criterion(batch_pred, batch_labels)
            batch_loss_val = batch_loss.item()

            if i % print_iter == 0:
                print('[{}/{}] batch loss:{}'.format(i, iter_per_epoch, batch_loss_val))

            mean_loss = mean_loss + batch_loss_val  # 累加所有批次的平均损失. item()的意思是取数值,因为该变量不是一个tensor

            batch_loss.backward()  # 反向传播损失,更新模型参数
            optimizer.step()  # 更新学习率

        mean_loss = mean_loss / (i + 1)  # 损失和求均,为当前epoch的损失

        with torch.no_grad():
            all_labels = []
            all_preds = []
            for i, (batch_data, batch_labels) in enumerate(test_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
                batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
                batch_pred = model(batch_data)  # 前向传播,获得网络的输出

                preds = batch_pred.squeeze().cpu().numpy().argmax(axis=1)  # 依次进行: 降维,转为cpu张量,转为np,求每一行最大值的索引
                labels = batch_labels.squeeze().cpu().numpy()

                all_labels.extend(list(labels))
                all_preds.extend(list(preds))

        f1 = f1_score(all_labels, all_preds, average='weighted')
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), save_dir + 'best.pth')
            print('Epoch:{}, train loss:{}, saved best model f1:{}'.format(epoch, mean_loss, f1))
        else:
            print('Epoch:{}, train loss:{}'.format(epoch, mean_loss))


def visualize_tensor_in_out(input, output, save_path):
    with torch.no_grad():
        input = input.squeeze().cpu()
        output = output.squeeze().cpu()

    plt.plot(input, 'b')
    plt.plot(output, 'r')
    plt.fill_between(np.arange(256), output, input, color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.savefig(save_path)
    plt.cla()


def plot_confusion_matrix(confusion, save_path):
    # 热度图，后面是指定的颜色块，可设置其他的不同颜色
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(class_names))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # axis = [str(i) for i in range(num_classes)]
    plt.xticks(indices, class_names)
    plt.yticks(indices, class_names)

    plt.colorbar()

    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            plt.text(first_index, second_index, confusion[first_index][second_index])
    # 在matlab里面可以对矩阵直接imagesc(confusion)

    plt.savefig(save_path)
    plt.cla()


def eval():
    model_path = save_dir + 'best.pth'
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        all_labels = []
        all_preds = []
        for i, (batch_data, batch_labels) in enumerate(test_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
            batch_pred = model(batch_data)  # 前向传播,获得网络的输出

            preds = batch_pred.squeeze().cpu().numpy().argmax(axis=1)  # 依次进行: 降维,转为cpu张量,转为np,求每一行最大值的索引
            labels = batch_labels.squeeze().cpu().numpy()

            print('preds', preds)
            print('label', labels)
            print()

            all_labels.extend(list(labels))
            all_preds.extend(list(preds))

    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='weighted')
    r = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf = confusion_matrix(all_labels, all_preds)
    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1}
    print(metrics)

    plot_confusion_matrix(conf, save_dir + 'confusion_matrix.png')


def demo():
    test_dataset = UCDataset('../UCMerced_LandUse/test64', image_size=64)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Finish loading dataset, begin testing.')

    model_path = save_dir + 'best.pth'
    model.load_state_dict(torch.load(model_path))

    for i, (batch_data, batch_labels) in enumerate(test_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
        with torch.no_grad():
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
            batch_pred = model(batch_data)  # 前向传播,获得网络的输出

            pred = batch_pred.cpu().numpy().argmax(axis=1)  # 依次进行: 降维,转为cpu张量,转为np,求每一行最大值的索引

            lab = batch_labels.squeeze().cpu().item()
            pre = pred[0]
            if lab == pre:
                flag = True
                color = 'green'
            else:
                flag = False
                color = 'red'

            text = 'Label is:{}, Prediction is:{}, Recognized:{}'.format(class_id2name[lab], class_id2name[pre], flag)
            print(text)

            # plt.title(text)
            # plt.plot(np.arange(256), batch_data.squeeze().cpu().numpy(), color=color)
            # plt.show()


if __name__ == '__main__':
    train()

    eval()

    # demo()