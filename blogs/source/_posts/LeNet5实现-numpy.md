---
title: LeNet5实现-numpy
categories:
  - [算法, 深度学习]
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - LeNet-5
  - python
  - numpy
abbrlink: c300ea0f
date: 2019-05-27 19:20:06
---

参考：

[卷积神经网络推导-批量图片矩阵计算](https://www.zhujian.tech/posts/ab1e719c.html#more)

[im2col解析1](https://www.zhujian.tech/posts/cc37c46b.html#more)

使用`numpy`实现`LeNet-5`网络，参考[toxtli/lenet-5-mnist-from-scratch-numpy](https://github.com/toxtli/lenet-5-mnist-from-scratch-numpy)模块化网络层

完整代码：[zjZSTU/PyNet](https://github.com/zjZSTU/PyNet)

## 卷积层

前向传播过程中各变量大小变化如下：

|  变量  |                 大小                 |
|:------:|:------------------------------------:|
|  input |               [N,C,H,W]              |
|    a   | [N\*out_h\*out_w, C\*filter_h\*filter_w] |
|    W   |    [C\*filter_h*filter_w, filter_num]   |
|    b   |            [1, filter_num]           |
|    z   |     [N\*out_h\*out_w, filter_num]    |
| output | [N, filter_num, out_h,out_w]         |

有以下注意：

* 需要将输入参数`input`转换成`2`维行向量矩阵`a`

    ```
    im2row_indices(x, field_height, field_width, padding=1, stride=1)
    ```
* 需要将`2`维矩阵`z`转换成`4`维数据体`output`

    ```
    conv_fc2output(inputs, batch_size, out_height, out_width):
    ```

反向传播过程中各变量梯度大小变化如下：

|   变量  |                 大小                 |
|:-------:|:------------------------------------:|
| doutput |     [N, filter_num, out_h,out_w]     |
|    dz   |     [N\*C\*out_h\*out_w, filter_num]    |
|    dW    |    [filter_h\*filter_w, filter_num]   |
|    db    |            [1, filter_num]           |
|    da   | [N\*out_h\*out_w, C\*filter_h\*filter_w] |
|  dinput |               [C,N,H,W]              |

有以下注意：

* 需要将`4`维输入`doutput`转换成`2`维梯度矩阵`da`

    ```
    conv_output2fc(inputs):
    ```
* 需要将`2`维梯度矩阵`da`转换成`4`维梯度数据体`dinput`

    ```
    row2im_indices(rows, x_shape, field_height=3, field_width=3, padding=1, stride=1, isstinct=False):
    ```

## 池化层

池化层和卷积层的最大差别在于每次池化层操作仅对单个激活图进行

前向传播过程中各变量大小变化如下：

|  变量 |                 大小                 |
|:-----:|:------------------------------------:|
| input |             [N, C, H, W]             |
|   a   | [N\*C\*out_h\*out_w, filter_h\*filter_w] |
|   z   |           [N\*C\*out_h\*out_w]          |
| onput |         [N, C, out_h, out_w]         |

有以下注意：

* 需要将输入参数`input`转换成`2`维行向量矩阵`a`

    ```
    pool2row_indices(x, field_height, field_width, stride=1):
    ```
* 需要将`1`维矩阵`z`转换成`4`维数据体`output`

    ```
    pool_fc2output(inputs, batch_size, out_height, out_width):
    ```

反向传播过程中各变量大小变化如下：

|   变量  |                 大小                 |
|:-------:|:------------------------------------:|
| doutput |         [N, C, out_h, out_w]         |
|    dz   |           [N\*C\*out_h\*out_w]          |
|    da   | [N\*C\*out_h\*out_w, filter_h\*filter_w] |
|  dinput |             [N, C, H, W]             |

有以下注意：

* 需要将`4`维输入`doutput`转换成`1`维矩阵`dz`

    ```
    pool_output2fc(inputs):
    ```
* 需要将`2`维梯度矩阵`da`转换成`4`维梯度数据体`dinput`

    ```
    row2pool_indices(rows, x_shape, field_height=2, field_width=2, stride=2, isstinct=False):
    ```

## 层定义

基本层定义分`3`部分功能：

1. 初始化
2. 前向传播
3. 反向传播

```
class Layer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass
```

对于有参数的层额外实现更新参数、获取参数和设置参数函数

```
def update(self, lr=1e-3, reg=1e-3):
    self.fc2.update(lr, reg)
    self.fc1.update(lr, reg)

def get_params(self):
    return {'fc1': self.fc1.get_params(), 'fc2': self.fc2.get_params()}

def set_params(self, params):
    self.fc1.set_params(params['fc1'])
    self.fc2.set_params(params['fc2'])
```

层实现代码：[ PyNet/nn/layers.py ](https://github.com/zjZSTU/PyNet/blob/master/nn/layers.py)

## 网络定义

网络实现以下功能：

1. 初始化
2. 前向传播
3. 反向传播
4. 参数更新
5. 获取参数
6. 设置参数

```
class Net(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass

    @abstractmethod
    def update(self, lr=1e-3, reg=1e-3):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass
```

网络实现代码：[ PyNet/nn/nets.py ](https://github.com/zjZSTU/PyNet/blob/master/nn/nets.py)

`LeNet-5`定义如下

```
class LeNet5(Net):
    """
    LeNet-5网络
    """

    def __init__(self):
        self.conv1 = Conv2d(1, 5, 5, 6, stride=1, padding=0)
        self.conv2 = Conv2d(6, 5, 5, 16, stride=1, padding=0)
        self.conv3 = Conv2d(16, 5, 5, 120, stride=1, padding=0)

        self.maxPool1 = MaxPool(2, 2, 6, stride=2)
        self.maxPool2 = MaxPool(2, 2, 16, stride=2)
        self.fc1 = FC(120, 84)
        self.fc2 = FC(84, 10)

        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.relu4 = ReLU()

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        x = self.relu1(self.conv1(inputs))
        x = self.maxPool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxPool2(x)
        x = self.relu3(self.conv3(x))
        # (N, C, 1, 1) -> (N, C)
        x = x.squeeze()
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x

    def backward(self, grad_out):
        da6 = self.fc2.backward(grad_out)

        dz6 = self.relu4.backward(da6)
        da5 = self.fc1.backward(dz6)
        # [N, C] -> [N, C, 1, 1]
        N, C = da5.shape[:2]
        da5 = da5.reshape(N, C, 1, 1)

        dz5 = self.relu3.backward(da5)
        da4 = self.conv3.backward(dz5)

        dz4 = self.maxPool2.backward(da4)

        dz3 = self.relu2.backward(dz4)
        da2 = self.conv2.backward(dz3)

        da1 = self.maxPool1.backward(da2)
        dz1 = self.relu1.backward(da1)

        self.conv1.backward(dz1)

    def update(self, lr=1e-3, reg=1e-3):
        self.fc2.update(lr, reg)
        self.fc1.update(lr, reg)
        self.conv3.update(lr, reg)
        self.conv2.update(lr, reg)
        self.conv1.update(lr, reg)

    def get_params(self):
        out = dict()
        out['conv1'] = self.conv1.get_params()
        out['conv2'] = self.conv2.get_params()
        out['conv3'] = self.conv3.get_params()

        out['fc1'] = self.fc1.get_params()
        out['fc2'] = self.fc2.get_params()

        return out

    def set_params(self, params):
        self.conv1.set_params(params['conv1'])
        self.conv2.set_params(params['conv2'])
        self.conv3.set_params(params['conv3'])

        self.fc1.set_params(params['fc1'])
        self.fc2.set_params(params['fc2'])
```

## 保存和加载参数

将参数保存成文件，同时能够从文件中加载参数，使用`python`的`pickle`模块，参数以字典形式保存

```
def save_params(params, path='params.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(params, f, -1)


def load_params(path='params.pkl'):
    with open(path, 'rb') as f:
        param = pickle.load(f)
    return param
```

完整代码：[ PyNet/nn/net_utils.py ](https://github.com/zjZSTU/PyNet/blob/master/nn/net_utils.py)

## mnist数据

参考：[mnist数据集](https://www.zhujian.tech/posts/ba2ca878.html#more)

将`mnist`数据集下载解压后，加载过程中完成以下步骤：

1. 转换成`(32,32)`大小
2. 转换维数顺序：`[H, W, C] -> [C, H, W]`

完整代码：[ PyNet/src/load_mnist.py ](https://github.com/zjZSTU/PyNet/blob/master/src/load_mnist.py)

加载完成后还需要进行数据标准化，因为图像取值为`[0,255]`，参考`pytorch`使用，简易操作如下：

```
# 标准化
x_train = x_train / 255.0 - 0.5
x_test = x_test / 255.0 - 0.5
```

## LeNet-5训练

训练参数如下：

1. 学习率`lr = 1e-3`
2. 正则化强度`reg = 1e-3`
3. 批量大小`batch_size = 128`
4. 迭代次数`epochs = 1000`

```
    net = LeNet5()
    criterion = CrossEntropyLoss()

    loss_list = []
    range_list = np.arange(0, x_train.shape[0] - batch_size, step=batch_size)
    for i in range(epochs):
        total_loss = 0
        num = 0
        start = time.time()
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train[j:j + batch_size]

            scores = net(data)
            loss = criterion(scores, labels)
            total_loss += loss
            num += 1

            grad_out = criterion.backward()
            net.backward(grad_out)
            net.update(lr=lr, reg=reg)
        end = time.time()
        print('one epoch need time: %.3f' % (end - start))
        print('epoch: %d loss: %f' % (i + 1, total_loss / num))
        loss_list.append(total_loss / num)
        # draw(loss_list)
        if i % 50 == 49:
            path = 'lenet5-epochs-%d.pkl' % (i + 1)
            params = net.get_params()
            save_params(params, path=path)
            test_accuracy = compute_accuracy(x_test, y_test, net, batch_size=batch_size)
            print('epochs: %d test_accuracy: %f' % (i + 1, test_accuracy))
            print('loss: %s' % loss_list)
```

完整代码：[ PyNet/src/lenet-5_test.py ](https://github.com/zjZSTU/PyNet/blob/master/src/lenet-5_test.py)

## 训练结果

训练时间

|                  计算机硬件                  	| 单次迭代时间 	|
|:--------------------------------------------:	|:------------:	|
| 12核 Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz 	|    约166秒   	|

训练结果

| 训练集精度 	| 测试集精度 	|
|:----------:	|:----------:	|
|   99.99%   	|   99.04%   	|

![](/imgs/LeNet5实现-numpy/lenet5-loss.png)


