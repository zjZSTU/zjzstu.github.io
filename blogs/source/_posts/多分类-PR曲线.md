---
title: '[多分类]PR曲线'
categories:
  - - 算法
    - 评价标准
    - PR曲线
  - - 编程
    - 编程语言
  - - 编程
    - 代码库
tags:
  - AP
  - python
  - sklearn
abbrlink: 2bbcad17
date: 2020-01-11 22:25:28
---

参考：

[ [二分类]PR曲线](https://zhujian.tech/posts/bca792b4.html)

[[多分类]ROC曲线](https://zhujian.tech/posts/48526d13.html)

计算多分类任务的`PR`曲线

## 解析

参考：[Mean average precision](https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision)

计算多分类任务的`PR`曲线面积`AP`，通常使用微平均方式（`micro`），即累加所有的标签的混淆矩阵，统一计算精确度和召回率

## python

`sklearn`库提供了函数[sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)计算`AP`

```
def average_precision_score(y_true, y_score, average="macro", pos_label=1, sample_weight=None):
```

利用函数[sklearn.metrics.precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)计算微平均方式的精确度和召回率

```
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
```

## 示例

参考：[Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html?highlight=precision%20recall#precision-recall)

使用`iris`数据集和单层神经网络进行`PR`曲线绘制

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   multi-pr-nn.py
@time:   2020-01-11
"""

import numpy as np
import matplotlib.pyplot as plt
from nn_classifier import NN
from sklearn.preprocessing import label_binarize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def load_data():
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    n_classes = 3

    # 数据标准化
    x_train = X_train.astype(np.float64)
    x_test = X_test.astype(np.float64)
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    eps = 1e-8
    x_train = (x_train - mu) / np.sqrt(np.maximum(var, eps))
    x_test = (x_test - mu) / np.sqrt(np.maximum(var, eps))

    # 定义分类器，训练和预测
    classifier = NN(None, input_dim=4, num_classes=3)
    classifier.train(x_train, y_train, num_iters=100, batch_size=8, verbose=True)
    res_labels, y_score = classifier.predict(x_test)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Binarize the output 将类别标签二值化
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # 计算微定义AP
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    # plt.figure()
    # plt.step(recall['micro'], precision['micro'], where='post')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    # plt.show()

    # setup plot details
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.15)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()
```

实现结果如下：

```
iteration 0 / 100: loss 1.080253
iteration 10 / 100: loss 0.848652
iteration 20 / 100: loss 0.703864
iteration 30 / 100: loss 0.611669
iteration 40 / 100: loss 0.549452
iteration 50 / 100: loss 0.504679
iteration 60 / 100: loss 0.470638
iteration 70 / 100: loss 0.443626
iteration 80 / 100: loss 0.421484
iteration 90 / 100: loss 0.402878
Average precision score, micro-averaged over all classes: 0.92
```

![](/imgs/multi-pr/multi-pr.png)