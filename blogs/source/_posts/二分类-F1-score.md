---
title: '[二分类]F1-score'
categories:
  - [算法, 评价标准, PR曲线]
tags:
  - F1 score
abbrlink: 50c7d392
date: 2020-01-10 20:01:04
---

参考：

[F1 score](https://en.wikipedia.org/wiki/F1_score)

[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

$F_{1} score$可以解释为精确性和召回率的加权平均值，相当于精确率和召回率的综合评价指标

***当前着重于二分类`F1 score`***

## 计算

精确率和召回率的计算参考[[二分类]PR曲线](https://zhujian.tech/posts/bca792b4.html)。$F_{1} score$的计算公式如下：

$$
F_{1} = \frac {2}{recall^{-1} + precision^{-1}} = 2\cdot \frac {precision\cdot recall}{precision + recall}
$$

$F_{1}$取值为`[0, 1]`，其中数值为`1`表示实现了最好的精确率和召回率，数值为`0`表示性能最差

## python

参考：[sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html?highlight=f1%20score#sklearn.metrics.f1_score)

`Python`库`Sklearn`实现了$F_{1} score$的计算

```
def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
```

该函数返回二元分类中正样本的`F1 score`值

* `y_true`：一维数组，表示正样本标签
* `y_pred`：一维数组，表示分类器预测类别
* `pos_label`：字符串或者数值，表示正样本类标签，默认为`1`

## 示例

参考[[二分类]PR曲线](https://zhujian.tech/posts/bca792b4.html)实现二元数据集的提取，分类器的训练和预测。`F1-score`计算如下：

```
from sklearn.metrics import f1_score

classifier = LogisticClassifier()
classifier.train(x_train, train_labels)
res_labels, scores = classifier.predict(x_test)

f1 = f1_score(test_labels, res_labels)
print(f1)
```