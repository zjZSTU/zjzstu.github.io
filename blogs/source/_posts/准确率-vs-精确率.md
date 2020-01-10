---
title: 准确率 vs. 精确率
categories:
  - - 算法
    - 评价标准
tags:
  - 准确率
  - 精确率
abbrlink: 5b516f3c
date: 2020-01-10 19:40:09
---

参考：[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

准确率和精确率是常用的算法评价标准，但是其定义略有差别

* 准确率（`Accuracy`）：预测正确的样本占所有样本的比率

$$
ACC = \frac {TP + TN}{P+N} = \frac {TP+TN}{TP+TN+FP+FN}
$$

* 精度率（`Precision`，也称为`PPV,  positive predictive value`）：预测正确的正样本占原先正样本集的比率

$$
PPV=\frac {TP}{TP+FP}
$$