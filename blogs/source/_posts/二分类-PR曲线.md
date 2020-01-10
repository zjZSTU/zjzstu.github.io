---
title: "[二分类]PR曲线"
categories:
  - [算法, 评价标准, PR曲线]
  - - 编程
    - 编程语言
  - - 编程
    - 代码库
tags:
  - AP
  - 精确率
  - 召回率
  - python
  - sklearn
abbrlink: bca792b4
date: 2019-12-27 13:52:21
---

参考：

[Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)

[混淆矩阵](https://zhujian.tech/posts/74ea027a.html)

`PR`曲线是另一种衡量算法性能的评价标准，其使用精确度（`Precision`）和召回率（`Recall`）作为坐标系的基底

***本文着重于二分类的PR曲线***

## 精确度

参考：[Positive and negative predictive values](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)

精确度（`Precision`）也称为正预测值（`positive predictive value, PPV`），表示预测正确的正样本占整个实际正样本集的比率

$$
PPV = \frac {TP}{TP + FP}
$$

## 召回率

参考：[Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)

召回率（`Recall`）也称为敏感度（`sensitivity`）、真阳性率（`true positive rate, TPR`），表示预测正确的正样本占整个预测正样本集的比率

$$
TPR = \frac {TP}{TP + FN}
$$

## PR曲线

`PPV`和`TPR`两者都是对于预测正样本集的理解和衡量

* 高精度意味着算法预测结果能够更好的覆盖所有的正样本（也就是**查准率**），但也可能存在更多的假阴性样本
* 高召回率意味着算法预测结果中包含了更多的正样本（也就是**查全率**），但也可能导致正样本占实际正样本的比率不高（存在更多的假阳性样本）

`PR`曲线是一个图，其`y`轴表示精度，`x`轴表示召回率，通过在不同阈值条件下计算`(Recall, Precision)`数据对，绘制得到`PR`曲线

根据定义可知，最好的预测结果发生在右上角`(1,1)`，此时所有预测为真的样本均为实际正样本，没有正样本被预测为假

## 如何通过PR判断分类器性能 - AP

和`ROC`曲线类似，需要计算曲线下面积来评判分类器性能，称之为平均精度（`AP, average precision`）

$$
AP = \sum_{n}(R_{n} - R_{n-1})P_{n}
$$

点$(R_{n}, P_{n})$表示第$n$个阈值下的精度和召回率

## Python实现

`Python`库`Sklearn`提供了`PR`曲线的计算函数：

* [sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
* [sklearn.metrics.precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)

### average_precision_score

```
def average_precision_score(y_true, y_score, average="macro", pos_label=1,
                            sample_weight=None):
```

用于计算预测成绩的平均精度

* `y_true`：数组形式，二值标签
* `y_score`：目标样本的成绩
* `pos_label`：正样本标签，默认为`1`

```
import numpy as np
from sklearn.metrics import average_precision_score

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
average_precision_score(y_true, y_scores)
```

### precision_recall_curve

```
def precision_recall_curve(y_true, probas_pred, pos_label=None,
                           sample_weight=None):
```

计算不同概率阈值下的精确率和召回率

* `y_true`：数组形式，表示样本标签，如果不是`{-1,1}`或者`{0,1}`形式，那么属性`pos_label`应该指定
* `probas_pred`：预测置信度
* `pos_label`：正样本类，默认为`1`

返回`3`个数组，分别是精确率数组、召回率数组和阈值数组

```
import numpy as np
from sklearn.metrics import precision_recall_curve

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
```

### 计算最佳阈值

综合来看，就是最接近坐标`(1,1)`的点所对应的阈值就是最佳阈值

```
best_th = threshold[np.argmax(precision + recall)]
```

## 示例

参考[[二分类]ROC曲线](https://zhujian.tech/posts/71a847e.html)使用`Fashion-MNIST`数据集，分两种情况

1. `6000`个运动鞋+`6000`个短靴作为训练集
2. `1000`个运动鞋+`6000`个短靴作为训练集

### 测试１

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   2-pr.py
@time:   2020-01-10
"""

from mnist_reader import load_mnist
from lr_classifier import LogisticClassifier
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def get_two_cate(ratio=1.0):
    path = "/home/zj/data/fashion-mnist/fashion-mnist/data/fashion/"
    train_images, train_labels = load_mnist(path, kind='train')
    test_images, test_labels = load_mnist(path, kind='t10k')

    num_train_seven = np.sum(train_labels == 7)
    num_train_nine = np.sum(train_labels == 9)
    # print(num_train_seven, num_train_nine)

    num_test_seven = np.sum(test_labels == 7)
    num_test_nine = np.sum(test_labels == 9)
    # print(num_test_seven, num_test_nine)

    x_train_0 = train_images[(train_labels == 7)]
    x_train_1 = train_images[(train_labels == 9)]
    y_train_0 = train_labels[(train_labels == 7)]
    y_train_1 = train_labels[(train_labels == 9)]

    x_train = np.vstack((x_train_0[:int(ratio * num_train_seven)], x_train_1))
    y_train = np.concatenate((y_train_0[:int(ratio * num_train_seven)], y_train_1))
    x_test = test_images[(test_labels == 7) + (test_labels == 9)]
    y_test = test_labels[(test_labels == 7) + (test_labels == 9)]

    return x_train, (y_train == 9) + 0, x_test, (y_test == 9) + 0


def compute_accuracy(y, y_pred):
    num = y.shape[0]
    num_correct = np.sum(y_pred == y)
    acc = float(num_correct) / num
    return acc


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = get_two_cate()

    print(train_images.shape)
    print(test_images.shape)

    # cv2.imshow('img', train_images[100].reshape(28, -1))
    # cv2.waitKey(0)

    x_train = train_images.astype(np.float64)
    x_test = test_images.astype(np.float64)
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    eps = 1e-8
    x_train = (x_train - mu) / np.sqrt(np.maximum(var, eps))
    x_test = (x_test - mu) / np.sqrt(np.maximum(var, eps))

    classifier = LogisticClassifier()
    classifier.train(x_train, train_labels)
    res_labels, scores = classifier.predict(x_test)

    acc = compute_accuracy(test_labels, res_labels)
    print(acc)

    precision, recall, threshold = precision_recall_curve(test_labels, scores, pos_label=1)
    fig = plt.figure()
    plt.plot(precision, recall, label='PR')
    plt.legend()
    plt.show()

    best_th = threshold[np.argmax(precision + recall)]
    print(best_th)
    y_pred = scores > best_th + 0
    acc = compute_accuracy(test_labels, y_pred)
    print(acc)
```

训练结果如下：

![](/imgs/pr-lr/pr-logistic-regression.png)

```
(12000, 784)
(2000, 784)
0.9205                                                 # 阈值为0.5
0.45903893031121357
0.9285                                                 # 阈值为0.4590
```

通过寻找最佳阈值，使得最后的准确率增加了`0.8%`

### 测试2

```
train_images, train_labels, test_images, test_labels = get_two_cate(ratio=1.0 / 6)
```

训练结果如下：

![](/imgs/pr-lr/pr-logistic-regression-2.png)

```
(7000, 784)
(2000, 784)
0.871                                               # 阈值为0.5
0.33526167648147953
0.9215                                            # 阈值为0.3353
```

从结果可知，`PR`曲线同样能够在类别数目不平衡的情况下有效的评估分类器性能

![](/imgs/pr-lr/pr-compare.png)
