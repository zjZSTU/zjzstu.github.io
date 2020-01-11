---
title: '[多分类]ROC曲线'
categories:
  - - 算法
    - 评价标准
    - ROC曲线
  - - 编程
    - 编程语言
  - - 编程
    - 代码库
tags:
  - AUC
  - python
  - sklearn
abbrlink: 48526d13
date: 2020-01-11 19:21:03
---

参考：[[二分类]ROC曲线](https://zhujian.tech/posts/71a847e.html)

学习和使用多分类任务的`ROC`曲线

## 解析

参考：[[3.3.2.1. From binary to multiclass and multilabel](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel)]

通过`ROC`曲线能够有效评估算法的性能，默认情况下适用于二分类任务，在多分类任务中利用`one vs rest`方式计算各个类别的混淆矩阵，使用如下平均方式

1. `macro`：分别求出每个类，再进行算术平均
   - 优点：直观、易懂，并且方便实现
   - 缺点：实际情况下可能不同类别拥有不同的重要性，宏平均会导致计算结果受不常用类别的影响
2. `weighted`：加权累加每个类别
3. `micro`：全局计算。将所有混淆矩阵累加在一起，然后计算`TPR/FPR/AUC`
4. `samples`：适用于样本不平衡的情况，参考[详解sklearn的多分类模型评价指标](https://zhuanlan.zhihu.com/p/59862986)

## python

`sklearn`库提供了函数[sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)计算`ROC`曲线的`AUC`

```
def roc_auc_score(y_true, y_score, average="macro", sample_weight=None, max_fpr=None):
```

* `y_true`：大小为[n_samples]（仅表示正样本标签）或者`[n_samples, n_classes]`
* `y_score`：大小和`y_true`一致，表示预测成绩
* `average`：适用于多分类任务的平均方式
    - `micro`：微平均方式。求和所有类别的混淆矩阵，再计算`TPR/FPR`
    - `macro`：宏平均方式。计算各个类别的混淆矩阵，再计算平均值
    - `weighted`：加权平均
    - `samples`

## 示例

参考：[Receiver Operating Characteristic (ROC)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py)

### 数据集

使用`sklearn`库提供的`iris`数据集，每类`50`个样本，每个样本包含`4`个特征，共`3`类

```
def load_data():
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    return X_train, X_test, y_train, y_test
```

## 分类器

参考[神经网络分类器](https://zhujian.tech/posts/81a57a7.html)使用单层神经网络

## 实现

通过两种方式计算多分类任务的`macro AUC`和`micro AUC`，一是通过分别计算每类的`TRP/FPR`，再计算最后的`AUC`；二是直接调用函数`roc_auc_score`计算

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   multi-roc-nn.py
@time:   2020-01-11
"""

# -*- coding: utf-8 -*-

"""
@author: zj
@file:   2-roc.py
@time:   2020-01-10
"""

from mnist_reader import load_mnist
from nn_classifier import NN

import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


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
    # print(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Binarize the output 将类别标签二值化
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    # one vs rest方式计算每个类别的TPR/FPR以及AUC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # 微平均方式计算TPR/FPR，最后得到AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # 直接调用函数计算
    micro_auc = roc_auc_score(y_test, y_score, average='micro')

    lw = 2
    # plt.figure()
    # plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # 直接调用函数计算
    macro_auc = roc_auc_score(y_test, y_score, average='macro')

    print(roc_auc)
    print('micro auc:', micro_auc)
    print('macro auc:', macro_auc)

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
```

输出结果如下：

```
iteration 0 / 100: loss 1.078953
iteration 10 / 100: loss 0.822893
iteration 20 / 100: loss 0.663991
iteration 30 / 100: loss 0.563895
iteration 40 / 100: loss 0.497447
iteration 50 / 100: loss 0.450596
iteration 60 / 100: loss 0.415750
iteration 70 / 100: loss 0.388712
iteration 80 / 100: loss 0.367034
iteration 90 / 100: loss 0.349197
{0: 1.0, 1: 0.8764534883720929, 2: 0.9312, 'micro': 0.9399111111111111, 'macro': 0.9408812015503877}
micro auc: 0.9399111111111111
macro auc: 0.935884496124031
```

![](/imgs/multi-roc/multi-roc.png)