---
title: softmax分类器
categories:
  - [算法, 机器学习]
  - [算法, 分类器]
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - softmax
  - python
  - numpy
  - pandas
  - sklearn
  - matplotlib
abbrlink: e043b7fb
date: 2019-07-17 20:07:33
---

参考：[softmax回归](https://www.zhujian.tech/posts/2626bec3.html)

模仿[线性SVM分类器](https://www.zhujian.tech/posts/ebe205e.html#more)实现`softmax`分类器

## 分类器实现

```
# -*- coding: utf-8 -*-

# @Time    : 19-7-17 下午7:45
# @Author  : zj


import numpy as np


class SoftmaxClassifier(object):

    def __init__(self):
        self.W = None
        self.b = None

        self.lr = None
        self.reg = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        self.lr = learning_rate
        self.reg = reg

        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)
            self.b = np.zeros((1, num_classes))

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, dW, db = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * dW
            self.b -= learning_rate * db

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        scores = self.softmax(X)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        y_pred = np.argmax(probs, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg, delta=1):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        num_train = X_batch.shape[0]

        scores = self.softmax(X_batch)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        data_loss = -1.0 / num_train * np.sum(np.log(probs[range(num_train), y_batch]))
        reg_loss = 0.5 * reg * np.sum(self.W ** 2)

        loss = data_loss + reg_loss

        dscores = scores
        dscores[range(num_train), y_batch] -= 1
        dscores /= num_train
        dW = X_batch.T.dot(dscores) + reg * self.W
        db = np.sum(dscores)

        return loss, dW, db

    def softmax(self, x):
        """
        :param x: A numpy array of shape (N, D)
        :param w: A numpy array of shape (D)
        :param b: A numpy array of shape (1)
        :return: A numpy array of shape (N)
        """
        z = x.dot(self.W) + self.b
        z -= np.max(z, axis=1, keepdims=True)
        return z
```

## 实验

使用交叉验证方法寻找最优的学习率和正则化强度组合

```
# -*- coding: utf-8 -*-

# @Time    : 19-7-17 下午8:00
# @Author  : zj

from builtins import range
from softmax_classifier import SoftmaxClassifier
import pandas as pd
import numpy as np
import math
from sklearn import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_iris(iris_path, shuffle=True, tsize=0.8):
    """
    加载iris数据
    """
    data = pd.read_csv(iris_path, header=0, delimiter=',')

    if shuffle:
        data = utils.shuffle(data)

    species_dict = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['Species'] = data['Species'].map(species_dict)

    data_x = np.array(
        [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]).T
    data_y = np.array(data['Species'])

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def load_german_data(data_path, shuffle=True, tsize=0.8):
    data_list = pd.read_csv(data_path, header=None, sep='\s+')

    data_array = data_list.values
    height, width = data_array.shape[:2]
    data_x = data_array[:, :(width - 1)]
    data_y = data_array[:, (width - 1)]

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=shuffle)

    y_train = np.array(list(map(lambda x: 1 if x == 2 else 0, y_train)))
    y_test = np.array(list(map(lambda x: 1 if x == 2 else 0, y_test)))

    return x_train, x_test, y_train, y_test


def compute_accuracy(y, y_pred):
    num = y.shape[0]
    num_correct = np.sum(y_pred == y)
    acc = float(num_correct) / num
    return acc


def cross_validation(x_train, y_train, x_val, y_val, lr_choices, reg_choices, classifier=SoftmaxClassifier):
    results = {}
    best_val = -1  # The highest validation accuracy that we have seen so far.
    best_svm = None  # The LinearSVM object that achieved the highest validation rate.

    for lr in lr_choices:
        for reg in reg_choices:
            svm = classifier()

            svm.train(x_train, y_train, learning_rate=lr, reg=reg, num_iters=2000, batch_size=100, verbose=True)
            y_train_pred = svm.predict(x_train)
            y_val_pred = svm.predict(x_val)

            train_acc = np.mean(y_train_pred == y_train)
            val_acc = np.mean(y_val_pred == y_val)

            results[(lr, reg)] = (train_acc, val_acc)
            if best_val < val_acc:
                best_val = val_acc
                best_svm = svm

    return results, best_svm, best_val


def plot(results):
    # Visualize the cross-validation results
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results]  # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('validation accuracy')
    plt.show()


if __name__ == '__main__':
    iris_path = '/home/zj/data/iris-species/Iris.csv'
    x_train, x_test, y_train, y_test = load_iris(iris_path, shuffle=True, tsize=0.8)

    # data_path = '/home/zj/data/german/german.data-numeric'
    # x_train, x_test, y_train, y_test = load_german_data(data_path, shuffle=True, tsize=0.8)

    x_train = x_train.astype(np.double)
    x_test = x_test.astype(np.double)
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    eps = 1e-8
    x_train = (x_train - mu) / np.sqrt(var + eps)
    x_test = (x_test - mu) / np.sqrt(var + eps)

    lr_choices = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-2]
    reg_choices = [7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4]
    results, best_svm, best_val = cross_validation(x_train, y_train, x_test, y_test, lr_choices, reg_choices)

    plot(results)

    for k in results.keys():
        lr, reg = k
        train_acc, val_acc = results[k]
        print('lr = %f, reg = %f, train_acc = %f, val_acc = %f' % (lr, reg, train_acc, val_acc))

    print('最好的设置是： lr = %f, reg = %f' % (best_svm.lr, best_svm.reg))
    print('最好的测试精度： %f' % best_val)
```

批量大小为`100`，共迭代`2000`次

`Iris`数据集测试结果如下：

```
最好的设置是： lr = 0.005000, reg = 0.000008
最好的测试精度： 0.933333
```

![](/imgs/softmax分类器/iris_softmax.png)

德国信用卡数据集测试结果如下：

```
最好的设置是： lr = 0.050000, reg = 0.000075
最好的测试精度： 0.765000
```

![](/imgs/softmax分类器/iris_softmax.png)

`2000`次迭代后的测试结果，与`KNN`分类器和线性`SVM`分类器比较结果如下：

|     |  Iris  | German data |
|:---:|:------:|:-----------:|
| KNN | 93.33% |    73.5%    |
| SVM |   80%  |     75%     |
| SVM | 93.33% |    76.5%    |