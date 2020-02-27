---
title: '[译]Transfer Learning for Computer Vision Tutorial'
categories:
  - - 翻译
  - - 数据
    - 数据集
  - - 编程
    - 编程语言
  - - 编程
    - 代码库
tags:
  - python
  - pytorch
  - torchvision
abbrlink: c8566254
date: 2020-02-26 19:53:35
---

>In this tutorial, you will learn how to train a convolutional neural network for image classification using transfer learning. You can read more about the transfer learning at cs231n notes

在本教程中，您将学习如何使用迁移学习来训练用于图像分类的卷积神经网络。您可以在[cs231n notes](https://cs231n.github.io/transfer-learning/)上阅读更多关于迁移学习的信息

>Quoting these notes,
>>In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.

引用

实际上，很少有人从头开始训练整个卷积网络(随机初始化)，因为拥有足够大的数据集是相对罕见的。相反，通常在非常大的数据集(例如，包含120万张1000个类别的图像的ImageNet)上预处理ConvNet，然后将ConvNet用作感兴趣任务的初始化或固定特征提取器

>These two major transfer learning scenarios look as follows:
>* Finetuning the convnet: Instead of random initializaion, we initialize the network with a pretrained network, like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.
>* ConvNet as fixed feature extractor: Here, we will freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.

这两个主要的迁移学习场景如下:

* 微调网络：不使用随机初始化而是用一个预训练网络来初始化网络，就像在imagenet 1000数据集上训练的网络一样。剩下的训练看起来和往常一样
* 卷积网络作为固定的特征提取器：除了最后的全连接层外，将会冻结所有网络的权重。最后的全连接层将会被一个新的随机初始化的全连接层替代，并且仅训练该层

```
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
```

## Load Data

加载数据

>We will use torchvision and torch.utils.data packages for loading the data.

我们使用torchvision和torch.util.data包来加载数据

>The problem we’re going to solve today is to train a model to classify ants and bees. We have about 120 training images each for ants and bees. There are 75 validation images for each class. Usually, this is a very small dataset to generalize upon, if trained from scratch. Since we are using transfer learning, we should be able to generalize reasonably well.

要解决的问题是训练一个分类蚂蚁和蜜蜂的模型，分别有大约120张蚂蚁和蜜蜂的训练图片，并且每个类还有75张验证图像。通常来说这是一个非常小的数据集，如果从零开始训练的话会导致无法收敛。而使用迁移学习，应该能够得到相当好的泛化结果

>This dataset is a very small subset of imagenet.

本数据集是imagenet的一个非常小的子集

>Note: Download the data from here and extract it to the current directory.

注意：下载[数据](https://download.pytorch.org/tutorial/hymenoptera_data.zip)并提取到当前文件夹

```
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Visualize a few images

可视化图像

>Let’s visualize a few training images so as to understand the data augmentations.

可视化小部分训练图像，以便理解数据扩充

```
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

![](/imgs/译-transfer-learning/sphx_glr_transfer_learning_tutorial_001.png)

## Training the model

训练模型

>Now, let’s write a general function to train a model. Here, we will illustrate:
>* Scheduling the learning rate
>* Saving the best model

现在，让我们编写一个通用函数来训练一个模型。将实现以下功能：

* 调度学习率
* 保存最佳模型

>In the following, parameter scheduler is an LR scheduler object from torch.optim.lr_scheduler.

下面代码中，参数scheduler是来自于包torch.optim.lr_scheduler的学习率调度器对象

```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

## Visualizing the model predictions

可视化模型预测

>Generic function to display predictions for a few images

显示图像预测的通用函数

```
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```

## Finetuning the convnet

微调网络

>Load a pretrained model and reset final fully connected layer.

加载预处理模型并重置最后的全连接层

```
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

## Train and evaluate

训练和评估

>It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.

在CPU上大约需要15-25分钟。但是在GPU上只需要不到一分钟的时间

```
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

>Out

输出

```
Epoch 0/24
----------
train Loss: 0.5952 Acc: 0.7172
val Loss: 0.2687 Acc: 0.9150

Epoch 1/24
----------
train Loss: 0.7011 Acc: 0.7172
val Loss: 0.2945 Acc: 0.9216
...
...
Epoch 23/24
----------
train Loss: 0.2068 Acc: 0.9139
val Loss: 0.2128 Acc: 0.9150

Epoch 24/24
----------
train Loss: 0.3255 Acc: 0.8443
val Loss: 0.2181 Acc: 0.9085

Training complete in 1m 7s
Best val Acc: 0.921569
```

可视化模型

```
visualize_model(model_ft)
```

![](/imgs/译-transfer-learning/sphx_glr_transfer_learning_tutorial_002.png)

## ConvNet as fixed feature extractor

使用卷积网络作为固定的特征提取器

>Here, we need to freeze all the network except the final layer. We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().

下面操作中将冻结除最后一层以外的所有网络。通过设置requires_grad == False来冻结参数，这样梯度就不会向后计算

>Here, we need to freeze all the network except the final layer. We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().

>You can read more about this in the documentation here.

更多关于梯度计算的可参考[Autograd mechanics](https://pytorch.org/docs/master/notes/autograd.html)

```
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

## Train and evaluate

训练和评估

>On CPU this will take about half the time compared to previous scenario. This is expected as gradients don’t need to be computed for most of the network. However, forward does need to be computed.

与之前的场景相比，在CPU上这将花费大约一半的时间。这是可预期的，虽然大多数网络不需要计算梯度，但是向前操作确实需要计算

```
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

>Out:

输出：

```
Epoch 0/24
----------
train Loss: 0.5803 Acc: 0.6680
val Loss: 0.3703 Acc: 0.8235

Epoch 1/24
----------
train Loss: 0.4522 Acc: 0.7869
val Loss: 0.1773 Acc: 0.9346

Epoch 2/24
----------
train Loss: 0.4594 Acc: 0.8197
val Loss: 0.2089 Acc: 0.9216
...
...
Epoch 23/24
----------
train Loss: 0.2957 Acc: 0.8525
val Loss: 0.2206 Acc: 0.9281

Epoch 24/24
----------
train Loss: 0.3527 Acc: 0.8443
val Loss: 0.2230 Acc: 0.9477

Training complete in 0m 34s
Best val Acc: 0.954248
```

可视化模型

```
visualize_model(model_conv)

plt.ioff()
plt.show()
```

![](/imgs/译-transfer-learning/sphx_glr_transfer_learning_tutorial_003.png)

## Further Learning

进一步学习

>If you would like to learn more about the applications of transfer learning, checkout our Quantized Transfer Learning for Computer Vision Tutorial.

如果想要学习更多关于迁移学习，参考[Quantized Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)

