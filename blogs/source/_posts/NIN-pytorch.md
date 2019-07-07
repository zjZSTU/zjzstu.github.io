---
title: NIN-pytorch
categories:
  - [算法]
  - [编程]
tags:
  - 深度学习
  - python
  - pytorch
abbrlink: cab4035
date: 2019-06-21 15:10:48
---

`numpy`实现[NIN](https://www.zhujian.tech/posts/359ae103.html#more)模型，利用`cifar-10`、`cifar-100`和`mnist`数据集进行`MLPConv`和`GAP`的测试

完整实现：[zjZSTU/PyNet](https://github.com/zjZSTU/PyNet)

## GAP实现

参考[Global Average Pooling in Pytorch](https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721)，使用[torch.nn.AvgPool2d](https://pytorch.org/docs/stable/nn.html?highlight=avgpool2d#torch.nn.AvgPool2d)

>class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

输入数据体大小为$N\times C\times H_{in}\times W_{in}$，输出大小为$N\times C\times H_{out}\times W_{out}$，则

$$
H_{out} = \left \lfloor \frac {H_{in}+2\times padding[0] - kernelsize[0]} {stride[0]} \right \rfloor + 1
$$

$$
W_{out} = \left \lfloor \frac {W_{in}+2\times padding[1] - kernelsize[1]} {stride[1]} \right \rfloor + 1
$$

设核空间尺寸为输入数据体大小，即为全局平均池化层

测试代码如下：

```
>>> gap = nn.AvgPool2d(3)
>>> a = torch.arange(36.).reshape(2,2,3,3)
>>> a
tensor([[[[ 0.,  1.,  2.],
          [ 3.,  4.,  5.],
          [ 6.,  7.,  8.]],

         [[ 9., 10., 11.],
          [12., 13., 14.],
          [15., 16., 17.]]],


        [[[18., 19., 20.],
          [21., 22., 23.],
          [24., 25., 26.]],

         [[27., 28., 29.],
          [30., 31., 32.],
          [33., 34., 35.]]]])
>>> gap(a)
tensor([[[[ 4.]],

         [[13.]]],


        [[[22.]],

         [[31.]]]])
>>> gap(a).shape
torch.Size([2, 2, 1, 1])
>>> res = gap(a).reshape(2,2)
>>> res
tensor([[ 4., 13.],
        [22., 31.]])
>>> gap(a).view(2,2) # 或使用view函数
tensor([[ 4., 13.],
        [22., 31.]])
```

## NIN定义

参考：[ pytorch-nin-cifar10/original.py ](https://github.com/jiecaoyu/pytorch-nin-cifar10/blob/master/original.py)

```
class NIN(nn.Module):

    def __init__(self, in_channels=1, out_channels=10):
        super(NIN, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 160, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(160, 96, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d()
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 192, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(192, 192, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d()
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(192, 192, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(192, out_channels, (1, 1), stride=1, padding=0),
            nn.ReLU(),
        )

        self.gap = nn.AvgPool2d(8)

    def forward(self, inputs):
        x = self.features1(inputs)
        x = self.features2(x)
        x = self.features3(x)
        x = self.gap(x)

        return x.view(x.shape[0], x.shape[1])
```

## 测试

```
def train():
    train_loader, test_loader = vision.data.load_cifar10_pytorch(data_path, batch_size=batch_size, shuffle=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    net = models.pytorch.nin(in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # stepLR = StepLR(optimer, 100, 0.5)

    best_train_accuracy = 0.995
    best_test_accuracy = 0

    accuracy = vision.Accuracy()

    loss_list = []
    train_list = []
    for i in range(epochs):
        num = 0
        total_loss = 0
        start = time.time()
        # 训练阶段
        net.train()
        for j, item in enumerate(train_loader, 0):
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
        end = time.time()
        # stepLR.step()

        avg_loss = total_loss / num
        loss_list.append(float('%.8f' % avg_loss))
        print('epoch: %d time: %.2f loss: %.8f' % (i + 1, end - start, avg_loss))

        if i % 20 == 19:
            # 验证阶段
            net.eval()
            train_accuracy = accuracy.compute_pytorch(train_loader, net, device)
            train_list.append(float('%.4f' % train_accuracy))
            if best_train_accuracy < train_accuracy:
                best_train_accuracy = train_accuracy

                test_accuracy = accuracy.compute_pytorch(test_loader, net, device)
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy

            print('best train accuracy: %.2f %%   best test accuracy: %.2f %%' % (
                best_train_accuracy * 100, best_test_accuracy * 100))
            print(loss_list)
            print(train_list)
```

