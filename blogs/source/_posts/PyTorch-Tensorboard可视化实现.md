---
title: '[PyTorch]Tensorboard可视化实现'
categories:
  - [编程, 编程语言]
  - [编程, 代码库]
  - - 工具
tags:
  - python
  - pytorch
  - torchvision
  - tensorboard
abbrlink: eb6f2b71
date: 2019-12-11 14:56:00
---

参考：[Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#visualizing-models-data-and-training-with-tensorboard)

最新版本的`PyTorch 1.3`内置支持了[Tensorboard](https://github.com/tensorflow/tensorboard)，实现模型、数据以及训练可视化

## 安装

除了安装`PyTorch`以外，还需要额外安装`Tensorboard`

```
conda install tensorboard
```

## 启动

启动`tensorboard`，指定文件路径，`IP`地址以及端口号

```
$ tensorboard --logdir PATH --host ADDR --port PORT
```

* `logdir`：指定`Tensorflow event files`文件路径，通常设置为`runs`，会递归搜索`runs`文件夹内命名为`*tfevents.*`文件
* `host`：制定监听的主机名，默认为`localhost`
* `port`：监听端口，默认为`6006`

```
$ tensorboard --logdir=runs --host=192.168.0.112 --port=7878
TensorFlow installation not found - running with reduced feature set.
W1211 15:14:25.693824 140379677206272 plugin_event_accumulator.py:294] Found more than one graph event per run, or there was a metagraph containing a graph_def, as well as one or more graph events.  Overwriting the graph with the newest event.
W1211 15:14:25.694132 140379677206272 plugin_event_accumulator.py:322] Found more than one "run metadata" event with tag step1. Overwriting it with the newest event.
TensorBoard 2.0.0 at http://192.168.0.112:7878/ (Press CTRL+C to quit)
```

## 常见问题

### pytorch ImportError: TensorBoard logging requires TensorBoard with Python summary writer installed

参考：[pytorch ImportError: TensorBoard logging requires TensorBoard with Python summary writer installed](https://blog.csdn.net/wangweiwells/article/details/101565407)

安装`tensorboard`即可

### tensorboard shows a SyntaxError: can't assign to operator

在`JupyterLab`上启动`Tensorboard`，发现如上问题，参考[tensorboard shows a SyntaxError: can't assign to operator](https://stackoverflow.com/questions/45392902/tensorboard-shows-a-syntaxerror-cant-assign-to-operator)。解决方案：在命令前添加`!`即可

```
!tensorboard --logdir=runs
```

### can't assign to operator

在`PyCharm`启动`tensorboard`时出现上述错误，参考[运行tensorboard --logdir=log遇到的错误之can't assign to operator](https://blog.csdn.net/weixin_40292207/article/details/80672041)，新开一个命令行窗口启动即可

## 入口函数

[SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)类提供了一个高级`API`，用于在给定目录中创建事件文件并向其中添加摘要和事件。该对象异步更新文件内容，这样允许训练程序直接在训练过程中调用方法循环向文件中添加数据，而不会减慢训练速度

所有的`PyTorch`数据写入操作均通过`SummaryWriter`实现，声明如下：

```
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```

需要指定文件保存路径，通常设置为`runs`

**注意：后续的写入操作完成后调用`close`函数结束**

## add_image

增加图像数据到`Tensorboard`，使用函数[add_image](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image)

**注意：需要额外安装`pillow`**

```
def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
```

* `tag`：数据标识符
* `img_tensor：torch.Tensor`格式图像

使用如下：

```
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/test')

lena = cv2.imread('lena.jpg')
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
lena_tensor = torch.from_numpy(lena.transpose((2, 0, 1)))
print(lena_tensor.size())

# 写入
writer.add_image('lena', lena_tensor)
writer.close()
```

完成上述操作后，在`tensorboard`页面菜单栏选择`IMAGES`标签，会出现写入的`lena`图像

![](/imgs/tensorboard/add_image.png)

*可以在同一个标签上添加多个图像，在页面上拖动滑动条显示不同的图像*

## add_graph

调用函数[add_graph](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph)实现模型可视化

```
def add_graph(self, model, input_to_model=None, verbose=False):
```

* `model：torch.nn.Module`（待绘制的模型）
* `input_to_model`：模型输入数据，可输入单张图像（`torch.Tensor`）或者图像列表（`list of torch.Tensor`）
* `verbose`：是否在控制台详细打印图结构

使用如下：

```
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# 获取数据
# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
                                             download=True,
                                             train=True,
                                             transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
                                            download=True,
                                            train=False,
                                            transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())

## 写入模型
writer.add_graph(net, input_to_model=images,verbose=True)
writer.close()
```

完成上述操作后，在`tensorboard`页面菜单栏选择`GRAPHS`标签，会可视化模型，双击即可扩展模型细节

![](/imgs/tensorboard/add_graph.png)

![](/imgs/tensorboard/expand_model.png)

## add_embedding

调用函数[add_embedding](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph)实现高维数据可视化

```
def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
```

* `mat`：矩阵，每行表示一个数据点的特征向量，其大小为$(N,D)$
* `metadata`：字符串列表，表示标签
* `label_imgs`：相对于每个数据点的图像列表，其大小为$(N,C,H,W)$

测试如下：

```
import keyword
import torch

// 提取关键字
meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist # get some strings
// 取前100个
meta = meta[:100]

// 转换成标签
for i, v in enumerate(meta):
    meta[i] = v+str(i)

// 随机生成100张图像
label_img = torch.rand(100, 3, 10, 32)
// 缩放到(0,1)
for i in range(100):
    label_img[i]*=i/100.0

writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
writer.close()
```

完成上述操作后，在`tensorboard`页面菜单栏选择`PROJECTOR`标签，将高维数据投影到`3`维空间中

![](/imgs/tensorboard/add_embedding.png)

## add_scalar

调用函数[add_scalar](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar)或者[add_scalars](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalars)实现训练数据实时写入

```
def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
```

* `main_tag`：标识符
* `scalar_value`：浮点值（`float`）或者字符串（`str`）
* `tag_scalar_dict：dict`，以键值对的方式保存子标签和相应的值
* `global_step`：步长

实现如下：

```
r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
writer.close()
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.
```

点击菜单栏的`SCALARS`标签，能够显示实时的训练数据

![](/imgs/tensorboard/add_scalar.png)

将鼠标移动到图中，会显示不同阶段相应的训练结果

![](/imgs/tensorboard/expand_scalar.png)

## add_figure

[add_figure](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure)作用与`add_image`类似，均是显示图像到`tensorboard`页面，不同的是`add_figure`指定渲染`Matplotlib`图像

```
def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
```

* `tag`：标识符
* `figure：matplotlib.pyplot.figure`格式图像
* `global_step`：步长

实现如下：

```
import matplotlib.pyplot as plt

fig = plt.figure()
plt.imshow(lena)
writer.add_figure('plt', fig, 0)

fig = plt.figure()
plt.imshow(gray, cmap="Greys")
writer.add_figure('plt', fig, 1)
```

写入后即可在`IMAGES`页面查询

## add_pr_curve

调用函数[add_pr_curve](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure)绘制精度召回曲线（`precision-recall curve`）

```
def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):
```

* `tag`：数据标识符
* `labels`：真实标签。取值为二值标签
* `predictions`：元素被归为正确的概率，取值为0或1
* `global_step`：步长
* `num_thresholds`：绘制曲线的阈值数量

```
labels = np.random.randint(2, size=100)  # binary label
print(labels)
predictions = np.random.rand(100)
print(predictions)

writer.add_pr_curve('pr_curve', labels, predictions, 0)
writer.close()
```