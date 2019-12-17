---
title: KNN分类器
categories: 
- [算法, 机器学习]
- [算法, 分类器]
- [编程, 编程语言]
- [编程, 代码库]
tags:
  - K近邻
  - python
  - numpy
  - matplotlib
  - sklearn
  - pandas
abbrlink: 1ee29eaf
date: 2019-07-11 19:34:29
---

参考：

[Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits](http://cs231n.github.io/classification/)

最近重温`cs231n`课程，完成了课堂作业[assignment1](http://cs231n.github.io/assignments2019/assignment1/)，记录一下`KNN`分类器相关的概念以及实现，包括通用的分类器训练及测试过程

## 什么是KNN分类器

参考：[机器学习笔记十一之决策边界](https://www.devtalking.com/articles/machine-learning-11/)

`KNN(K-Nearest Neighbor)`分类器是最简单的分类器实现之一，其决策边界是非线性的。它通过比较测试图像和样本图像的像素差异，按差异值从低到高排序对应样本图像标签，选择前`K`个标签中出现次数最多的标签作为分类结果

常用的比较像素差异的方法有`L1/L2`范数，参考[范数](https://www.zhujian.tech/posts/ce0afb50.html)

### 优势和劣势

主要有`2`点优势：

1. 易于理解和实现
2. 不需要花费时间训练

主要有`3`点缺陷：

1. 分类器需要保存所有的训练数据，空间效率低下
2. 测试过程中测试图像需要和所有的训练图像进行比较，时间效率低下
3. 通过像素差异进行分类，对于偏移（`shift`）、遮挡（`messed up`）和亮度调节的泛化效果不高

## KNN分类器实现

实现`KNN`分类器，使用`L2`范数进行像素差异计算，默认执行最近邻分类（*K=1*）

```
# -*- coding: utf-8 -*-

# @Time    : 19-7-11 下午8:02
# @Author  : zj

from builtins import range
from builtins import object
import numpy as np


class KNN(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        dists = self._compute_distances(X)

        return self._predict_labels(dists, k=k)

    def _compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        temp_test = np.atleast_2d(np.sum(X ** 2, axis=1)).T
        temp_train = np.atleast_2d(np.sum(self.X_train ** 2, axis=1))
        temp_test_train = -2 * X.dot(self.X_train.T)

        dists = np.sqrt(temp_test + temp_train + temp_test_train)
        return dists

    def _predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            idxes = np.argsort(dists[i])
            closest_y = list(self.y_train[idxes][:k])

            nums = np.array([closest_y.count(m) for m in closest_y])
            y_pred[i] = closest_y[np.argmax(nums)]

        return y_pred
```

## 交叉验证

交叉验证（`cross-validation`）是模型训练过程中调试超参数$k$的很有效的手段，训练过程如下：

1. 将训练数据等分为$N$份
2. 按序取出其中一份作为验证集，其余数据重新作为训练集
3. 计算超参数$k$为某一值时的预测精度。这样，每个超参数$k$都会得到$N$个精度值
4. 平均$N$个精度值作为当前超参数$k$值的检测结果，比较不同$k$值下的检测精度，选取最好的$k$值

```
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

# 计算预测标签和验证集标签的精度
def compute_accuracy(y_test, y_test_pred):
    num_test = y_test.shape[0]
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    return accuracy

for k in k_choices:
    k_accuracies = []
    # 随机选取其中一份为验证集，其余为测试集
    for i in range(num_folds):
        x_folds = X_train_folds.copy()
        y_folds = y_train_folds.copy()
        
        x_vals = x_folds.pop(i)
        x_trains = np.vstack(x_folds)
        
        y_vals = y_folds.pop(i)
        y_trains = np.hstack(y_folds)
        
        classifier = KNearestNeighbor()
#         print(x_trains.shape)
#         print(y_trains.shape)
        classifier.train(x_trains, y_trains)
        
#         print(x_vals.shape)
        y_val_pred = classifier.predict(x_vals, k=k, num_loops=0)
        k_accuracies.append(compute_accuracy(y_vals, y_val_pred))
    k_to_accuracies[k] = k_accuracies

# print(k_to_accuracies)
# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
```

*通常设置训练集为$5$等分或$10$等分*

交叉验证方法的优点在于其得到的超参数具有最好的泛化效果，缺点在于训练时间长

## 实验

使用两个数据集分别测试`KNN`分类器，使用交叉验证方法训练最好的`KNN`分类器

完整代码如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-7-15 上午11:33
# @Author  : zj

from builtins import range
from classifier.knn_classifier import KNN
import pandas as pd
import numpy as np
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
    data_y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    return x_train, x_test, y_train, y_test


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


def cross_validation(x_train, y_train, k_choices, num_folds=5, Classifier=KNN):
    X_train_folds = np.array_split(x_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    # 计算预测标签和验证集标签的精度
    k_to_accuracies = {}
    for k in k_choices:
        k_accuracies = []
        # 随机选取其中一份为验证集，其余为测试集
        for i in range(num_folds):
            x_folds = X_train_folds.copy()
            y_folds = y_train_folds.copy()

            x_vals = x_folds.pop(i)
            x_trains = np.vstack(x_folds)

            y_vals = y_folds.pop(i)
            y_trains = np.hstack(y_folds)

            classifier = Classifier()
            classifier.train(x_trains, y_trains)

            y_val_pred = classifier.predict(x_vals, k=k)
            k_accuracies.append(compute_accuracy(y_vals, y_val_pred))
        k_to_accuracies[k] = k_accuracies

    return k_to_accuracies


def plot(k_choices, k_to_accuracies):
    # plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


if __name__ == '__main__':
    # iris_path = '/home/zj/data/iris-species/Iris.csv'
    # x_train, x_test, y_train, y_test = load_iris(iris_path, shuffle=True, tsize=0.8)

    data_path = '/home/zj/data/german/german.data-numeric'
    x_train, x_test, y_train, y_test = load_german_data(data_path, shuffle=True, tsize=0.8)

    x_train = x_train.astype(np.double)
    x_test = x_test.astype(np.double)
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    eps = 1e-8
    x_train = (x_train - mu) / np.sqrt(var + eps)
    x_test = (x_test - mu) / np.sqrt(var + eps)

    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100]
    k_to_accuracies = cross_validation(x_train, y_train, k_choices)

    # print(k_to_accuracies)
    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    plot(k_choices, k_to_accuracies)

    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    k = k_choices[np.argmax(accuracies_mean)]
    print('最好的k值是：%d' % k)

    # 测试集测试
    classifier = KNN()
    classifier.train(x_train, y_train)

    y_test_pred = classifier.predict(x_test, k=k)
    y_test_acc = compute_accuracy(y_test, y_test_pred)
    print('测试集精度为：%f' % y_test_acc)
```

数据集一：`Iris`数据集，共`4`个变量，`3`个类别。参考：[鸢尾数据集](https://www.zhujian.tech/posts/2626bec3.html)

测试结果：

```
最好的k值是：12
测试集精度为：0.933333
```

![](/imgs/KNN/iris.png)

数据集二：德国信用卡数据集，共`24`个变量，`2`个类别。参考：[german data](https://download.csdn.net/download/u012005313/11351036)

测试结果：

```
最好的k值是：20
测试集精度为：0.735000
```

![](/imgs/KNN/german_data.png)

*由于数据集过小，进行多次交叉验证才能得到比较好的结果*