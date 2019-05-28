---
title: LeNet5实现-pytorch
categories:
  - 编程
tags:
  - 深度学习
abbrlink: a2db6d6b
date: 2019-05-28 11:06:13
---

参考：[神经网络实现-pytorch](https://www.zhujian.tech/posts/5a77dbca.html#more)

完整代码：[ PyNet/pytorch/lenet5_test.py ](https://github.com/zjZSTU/PyNet/blob/master/pytorch/lenet5_test.py)

## 加载数据

`pytorch`提供模块`torchvision`，用于数据的加载、预处理和批量化

* `torchvision.datasets`内置类`MNIST`用于`mnist`数据集下载和加载
* `torchvision.transforms`对数据进行预处理
* `torchvision.DataLoader`对数据进行批量化

```
def load_mnist_data(batch_size=128, shuffle=False):
    data_dir = '/home/zj/data/'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_data_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader
```

## 网络定义

`LeNet-5`模型定义参考[卷积神经网络推导-单张图片矩阵计算](https://www.zhujian.tech/posts/3accb62a.html#more)

`torch.nn`模块实现了网络层类，包括卷积层（`Conv2d`）、最大池化层（`MaxPool2d`）、全连接层（`Linear`）和其他激活层

`torch.nn`模块提供`functional`类用于网络层类的实现

```
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=True)

        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.fc1 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc2 = nn.Linear(84, 10, bias=True)

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
```

## 训练

训练参数如下

1. 学习率`lr = 1e-3`
2. 批量大小`batch_size = 128`
3. 迭代次数`epochs = 500`

### 训练结果

训练时间

|                      CPU                      	|      GPU      	| 单次迭代时间 	|
|:---------------------------------------------:	|:-------------:	|:------------:	|
| 8核 Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz 	| GeForce 940MX 	|    约13秒    	|

迭代`500`次训练结果

| 训练集精度 	| 测试集精度 	|
|:----------:	|:----------:	|
|   99.40%   	|   98.63%   	|

![](/imgs/LeNet5实现-pytorch/pytorch_lenet5_mnist_accuracy.png)

![](/imgs/LeNet5实现-pytorch/pytorch_lenet5_mnist_loss.png)


