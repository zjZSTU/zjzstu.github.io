---
title: '[pytorch]训练一个简单的检测器'
abbrlink: 5bfa4e56
date: 2020-01-18 20:30:40
categories:
- [算法, 深度学习, 卷积神经网络]
- [数据, 数据集]
- [编程, 代码库]
- [编程, 编程语言]
tags:
- LeNet-5
- AlexNet
- image localization dataset
- pytorch
- python
---

学习边框回归的概念时，发现一篇自定义检测器的文章

* 原文：[Getting Started With Bounding Box Regression In TensorFlow](https://towardsdatascience.com/getting-started-with-bounding-box-regression-in-tensorflow-743e22d0ccb3)
* 中文：[目标检测之边框回归入门【Tensorflow】](http://blog.hubwiz.com/2019/09/16/bounding-box-regression/)

虽然题目写的是边框回归，但是里面没有讲解相关的概念，而是自定义了一个边框检测器，实现原理比较简单。看完之后感觉挺有趣的，之前也没有自己实现过检测器，原文使用`TensorFlow`实现，当前使用`PyTorch`进行复现

## 操作流程

1. 定位数据集
2. 自定义损失函数
3. 训练
4. 检测

## 定位数据集

使用一个简单的[图像定位数据集](https://zhujian.tech/posts/a2d65e1.html)：

>* 包含`3`类：`Cucumber`（黄瓜）、`Eggplant`（茄子）、`Mushroom`（蘑菇）
>* 每类共有超过`60`张的图像，大小固定为`(227, 277, 3)`，每张图像里有一个物体
>* 每张图像有一个对应的`xml`文件，格式和`PASCAL VOC`数据集一致，包含图像信息以及标注信息

## 自定义损失函数

使用`MSE(Mean Squared Error)`和[IoU(Intersection over Union)](https://zhujian.tech/posts/796ebd4e.html)作为损失函数

$$
loss(x, x^{'}) = MSE(x, x^{'}) + (1 - IoU(x, x^{'})) 
$$

`MSE`计算的是预测标签和真值标签之间的损失，`1- IoU`计算的是预测边框和真值边框之间的损失，这样能够保证模型计算结果能够检测指定类别以及得到相应的标注信息。所以训练标签包含了边框标注数据和类别信息（`one-hot`编码形式）

## 训练

* 自定义了数据集类`LocationDataSet`用于图像定位数据集的加载
* 自定义了损失函数类`MSE_IoU`用于损失函数的计算和求导
* 分别使用`LeNet-5`和`AlexNet`进行训练
* 使用随机梯度下降进行模型优化，初始学习率为`1e-3`，动量大小为`0.9`，使用`Nesterov`加速，共训练`100`轮

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   bounding.py
@time:   2020-01-15
"""

import os
import logging
import cv2
import numpy as np
import xmltodict
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import alexnet

logging.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', level=logging.DEBUG)

# LeNet-5
# input_dim = 32
# AlexNet
input_dim = 227


class LocationDataSet(Dataset):

    def __init__(self, root_dir, train=True, transform=None, input_dim=1):
        """
        自定义数据集类，加载定位数据集
        1. 训练部分，加载编码前50图像和标记数据
        2. 测试部分，加载编码50之后图像和标记数据
        :param root_dir:
        :param train:
        :param transform:
        """
        cates = ['cucumber', 'eggplant', 'mushroom']
        class_binary_label = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.train = train
        self.transform = transform

        self.imgs = []
        self.bboxes = []
        self.classes = []

        for cate_idx in range(3):
            if self.train:
                for i in range(1, 51):
                    img, bndbox, class_name = self._get_item(root_dir, cates[cate_idx], i)
                    bndbox = bndbox / input_dim

                    self.imgs.append(img)
                    self.bboxes.append(np.hstack((bndbox, class_binary_label[cate_idx])))
                    self.classes.append(class_name)
            else:
                for i in range(51, 61):
                    img, bndbox, class_name = self._get_item(root_dir, cates[cate_idx], i)
                    bndbox = bndbox / input_dim

                    self.imgs.append(img)
                    self.bboxes.append(np.hstack((bndbox, class_binary_label[cate_idx])))
                    self.classes.append(class_name)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        return sample, torch.Tensor(self.bboxes[idx]).float()

    def __len__(self):
        return len(self.imgs)

    def _get_item(self, root_dir, cate, i):
        img_path = os.path.join(root_dir, '%s_%d.jpg' % (cate, i))
        img = cv2.imread(img_path)

        xml_path = os.path.join(root_dir, '%s_%d.xml' % (cate, i))
        x = xmltodict.parse(open(xml_path, 'rb'))
        bndbox = x['annotation']['object']['bndbox']
        bndbox = np.array(
            [float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(bndbox['ymax'])])

        return img, bndbox, x['annotation']['object']['name']


def load_data():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_dim),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    root_dir = './data/image-localization-dataset/training_images/'
    train_dataset = LocationDataSet(root_dir, train=True, transform=transform, input_dim=input_dim)
    test_dataset = LocationDataSet(root_dir, train=False, transform=transform, input_dim=input_dim)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader


class MSE_IoU(nn.Module):

    def calculate_iou(self, target_boxes, pred_boxes):
        # 计算重叠区域的左上角和右下角坐标
        x_min = torch.max(target_boxes[:, 0], pred_boxes[:, 0])
        y_min = torch.max(target_boxes[:, 1], pred_boxes[:, 1])
        x_max = torch.min(target_boxes[:, 2], pred_boxes[:, 2])
        y_max = torch.min(target_boxes[:, 3], pred_boxes[:, 3])
        # 计算交集面积
        intersection = torch.max(torch.zeros(x_max.shape).cuda(), x_max - x_min) \
                       * torch.max(torch.zeros(y_max.shape).cuda(), y_max - y_min)

        # 计算两个边界框面积
        boxAArea = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        boxBArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

        iou = intersection / (boxAArea + boxBArea - intersection)
        return iou

    def forward(self, target_boxes, pred_boxes):
        mseloss = nn.MSELoss().forward(target_boxes, pred_boxes)
        iouloss = torch.mean(1 - self.calculate_iou(target_boxes, pred_boxes))

        return mseloss + iouloss


class LeNet5(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=True)

        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.fc1 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc2 = nn.Linear(84, num_classes, bias=True)

    def forward(self, input):
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def compute_accuracy(loader, net, device):
    total_accuracy = 0
    num = 0
    for item in loader:
        data, labels = item
        data = data.to(device)
        labels = labels.to(device)

        scores = net.forward(data)
        predicted = torch.nn.functional.one_hot(torch.argmax(scores[:, 4:7], dim=1), num_classes=3)
        total_accuracy += torch.mean((predicted == labels[:, 4:7]).float()).item()
        num += 1
    return total_accuracy / num


if __name__ == '__main__':
    train_dataloader, test_dataloader = load_data()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_classes = 7
    # net = LeNet5(in_channels=3, num_classes=num_classes).to(device)
    net = alexnet(num_classes=num_classes).to(device)
    criterion = MSE_IoU().to(device)
    # optimer = optim.Adam(net.parameters(), lr=1e-3)
    optimer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    logging.info("开始训练")
    epoches = 100
    for i in range(epoches):
        num = 0
        total_loss = 0
        for j, item in enumerate(train_dataloader, 0):
            data, labels = item
            data = data.to(device)
            labels = labels.to(device)

            scores = net.forward(data)
            loss = criterion.forward(scores, labels)
            total_loss += loss.item()

            optimer.zero_grad()
            loss.backward()
            optimer.step()
            num += 1
        avg_loss = total_loss / num
        logging.info('epoch: %d loss: %.6f' % (i + 1, total_loss / num))
        train_accuracy = compute_accuracy(train_dataloader, net, device)
        test_accuracy = compute_accuracy(test_dataloader, net, device)
        logging.info('train accuracy: %f test accuracy: %f' % (train_accuracy, test_accuracy))

    # torch.save(net.state_dict(), './model/LeNet-5.pth')
    torch.save(net.state_dict(), './model/AlexNet.pth')

    img, label = test_dataloader.dataset.__getitem__(10)
    img = img.unsqueeze(0).to(device)
    print(img.shape)
    print(label)
    scores = net.forward(img)
    print(scores)
```

使用`AlexNet`训练日志如下：

```
2020-01-18 21:03:25,738 box-detector.py[line:230] INFO epoch: 98 loss: 0.225710
2020-01-18 21:03:26,068 box-detector.py[line:234] INFO train accuracy: 0.894737 test accuracy: 0.854167
2020-01-18 21:03:26,737 box-detector.py[line:230] INFO epoch: 99 loss: 0.224704
2020-01-18 21:03:27,065 box-detector.py[line:234] INFO train accuracy: 0.903509 test accuracy: 0.833333
2020-01-18 21:03:27,747 box-detector.py[line:230] INFO epoch: 100 loss: 0.230456
2020-01-18 21:03:28,076 box-detector.py[line:234] INFO train accuracy: 0.899123 test accuracy: 0.854167
torch.Size([1, 3, 227, 227])
tensor([0.2247, 0.2599, 0.8414, 0.7225, 0.0000, 1.0000, 0.0000])
tensor([[0.2072, 0.2580, 0.7714, 0.6773, 0.3503, 0.4326, 0.1010]],
       device='cuda:0', grad_fn=<AddmmBackward>)
```

模型保存在`./model/AlexNet.pth`中

## 检测

调用保存的模型进行标注检测，并比较相应的检测类别

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   draw_box.py
@time:   2020-01-18
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import alexnet

from box_detector import load_data
from box_detector import input_dim

if __name__ == '__main__':
    train_loader, test_loader = load_data()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = test_loader.dataset
    img, label = dataset.__getitem__(0)
    image = img.unsqueeze(0).to(device)
    label = label.unsqueeze(0)

    num_classes = 7
    # net = LeNet5(in_channels=3, num_classes=num_classes).to(device)
    net = alexnet(num_classes=num_classes).to(device)
    # net.load_state_dict(torch.load('./model/LeNet-5.pth'))
    net.load_state_dict(torch.load('./model/AlexNet.pth'))
    net.eval()

    scores = net.forward(image)
    print(scores)
    print(label)

    predict_cate = torch.argmax(scores[:, 4:7], dim=1)
    truth_cate = torch.argmax(label[:, 4:7], dim=1)
    print('predict: ' + str(predict_cate) + ' truth: ' + str(truth_cate))

    img = img * 0.5 + 0.5
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(227)
    ])

    truth_rect = label[:, :4] * input_dim
    predict_rect = scores[:, :4] * input_dim
    print(truth_rect)
    print(predict_rect)

    origin = np.array(transform(img), dtype=np.uint8)

    x_min, y_min, x_max, y_max = truth_rect.squeeze()[:4]
    cv2.rectangle(origin, (x_min, y_min), (x_max, y_max),
                  (0, 255, 0), thickness=2)
    x_min, y_min, x_max, y_max = predict_rect.squeeze()[:4]
    cv2.rectangle(origin, (x_min, y_min), (x_max, y_max),
                  (0, 0, 255), thickness=2)
    cv2.imwrite('box_detector.png', origin)
    # cv2.imshow('img', origin)
    # cv2.waitKey(0)
```

输出结果如下：

```
tensor([[  7.,  17., 220., 199.]])
tensor([[  2.9428,  16.3437, 221.9320, 199.8142]], device='cuda:0',
       grad_fn=<MulBackward0>)
```

![](/imgs/detector-location/box_detector.png)
