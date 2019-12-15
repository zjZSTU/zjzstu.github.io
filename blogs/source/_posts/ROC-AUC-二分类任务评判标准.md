---
title: '[ROC][AUC]二分类任务评判标准'
abbrlink: 887dcf29
date: 2019-12-13 14:55:38
categories:
  - - 评判标准
  - - 编程
tags:
- python
- sklearn
- ROC
- AUC
---

参考：

[Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)

[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

[Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

对于分类问题，最开始想到的评判标准就是检测准确率（`accuracy`），即样本检测类别和实际一致的数量占整个样本集的比率。进一步研究发现，还可以用更精细的标准来比较检测性能，学习步骤如下：

1. 正样本和负样本
2. `TP/FP/TN/FN`
3. `TPR/FPR/FDR/PPV/ACC`
4. `ROC/AUC`

## 正样本和负样本

在二分类问题中，将待识别的物体称为正样本（`positive case`），另外一个称为负样本（`negative case`）

## TP/FP/TN/FN

对数据进行检测，能够得到以下`4`种检测结果

* 预测结果是正样本
  * 实际是正样本，称为真阳性（`true positive`，简称`TP`）
  * 实际是负样本，称为假阴性（`false negative`, 简称`FN`）
* 预测结果是负样本
  * 实际是正样本，称为假阳性（`false positive`，简称`FP`）
  * 实际是负样本，成为真阴性（`true negative`, 简称`TN`）

也就是说，**根据实际情况**决定预测结果是阳性还是阴性；**根据预测结果和实际情况的比对**决定预测结果是真还是假

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="2">预测</th>
    <th class="tg-c3ow"></th>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">true</td>
    <td class="tg-c3ow">false</td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2">实际</td>
    <td class="tg-c3ow">positive</td>
    <td class="tg-c3ow">true positive(TP)</td>
    <td class="tg-c3ow">false positive(FP)</td>
    <td class="tg-c3ow">正样本个数=TP+FP</td>
  </tr>
  <tr>
    <td class="tg-c3ow">negative</td>
    <td class="tg-c3ow">false negative(FN)</td>
    <td class="tg-c3ow">true negative(TN)</td>
    <td class="tg-c3ow">负样本个数=FN+TN</td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">预测正样本个数=TP+FN</td>
    <td class="tg-c3ow">预测负样本个数=FP+TN</td>
    <td class="tg-c3ow"></td>
  </tr>
</table>

所以预测为真的样本数$P=TP+FN$，预测为负的样本数$N=FP+TN$

## TPR/FPR/FDR/PPV/ACC

### TPR/FPR

* 真阳性率（`TPR, true positive rate`），也称为敏感度（`sensitivity`）、召回率（`recall rate`）、检测率（`probability of detection`），其计算的是检测为真的正样本在整个检测为真的样本集中的比率

$$
TPR = \frac {TP}{P} = \frac {TP}{TP+FN}
$$

* 假阳性率（`FPR, false positive rate`），也称为误报率（`probability of false alarm`），其计算的是检测为假的正样本在整个检测为假的样本集中的比率

$$
FPR = \frac {FP}{N} = \frac {FP}{FP+TN}
$$

### FDR和PPV

* 漏检率（`FDR, false discovery rate`）计算的是假阳性样本占实际正样本集的比率

$$
FDR=\frac {FP}{TP+FP}
$$

* `PPV(positive predictive value)`，也称为精度（`precision`），其计算的是检测为真的正样本占整个实际正样本集的比率

$$
PPV=\frac {TP}{TP+FP}
$$

### ACC

`ACC(accuracy)`就是指正确率，指的是真阳性和真阴性占整个样本集的比率

$$
ACC = \frac {TP+TN}{P+N} = \frac {TP+TN}{TP+TN+FP+FN}
$$

## ROC/AUC

### ROC曲线

`ROC`曲线，全称是接受者操作特征曲线（`receiver operating characteristic curve`），它是一个二维图，用于表明分类器的检测性能

其`y`轴表示`TPR`，`x`轴表示`FPR`。通过在不同阈值条件下计算`(FPR, TPR)`数据对，绘制得到`ROC`曲线

![](/imgs/ROC-AUC/1024px-ROC_space-2.png)

`ROC`曲线描述了收益（`true positive`）和成本（`false positive`）之间的权衡。由上图可知

* 最好的预测结果发生在左上角`(0,1)`，此时所有预测为真的样本均为实际正样本，没有正样本被预测为假
* 对角线表示的是随机猜测（`random guess`）的结果，对角线上方的坐标点表示分类器的检测结果比随机猜测好

所以离左上角越近，表示预测效果越好，此时分类器的性能更佳

### AUC

参考：[如何理解机器学习和统计中的AUC？](https://www.zhihu.com/question/39840928?from=profile_question_card)

`AUC(area under the curve)`指的是`ROC`曲线图中曲线下方的面积。其表示概率值，表示当随机给定一个正样本和一个负样本，分类器输出该正样本为正的那个概率值比分类器输出该负样本为正的那个概率值要大的可能性

*通过计算AUC值，也可以判断出最佳阈值*

### 如何计算最佳阈值

参考：

[How to determine the optimal threshold for a classifier and generate ROC curve?](https://stats.stackexchange.com/questions/123124/how-to-determine-the-optimal-threshold-for-a-classifier-and-generate-roc-curve)

[Roc curve and cut off point. Python](https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python)

通过`ROC`图可知，`TPR`越大越好，`FPR`越小越好，所以只要能够得到不同阈值条件下的`TPR`和`FPR`，计算之间的差值，结果值最大的就是最佳阈值

```
thresh = thresholds[np.argmax(tpr - fpr)]
print(thresh)
```

## python实现

`sklean`库提供了多个函数用于`ROC/AUC`的计算，参考[3.3.2.14. Receiver operating characteristic (ROC)](https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc)

* [sklearn.metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
* [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
* [sklearn.metrics.auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html?highlight=auc#sklearn.metrics.auc)

### roc_curve

```
def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
```

* `y_true`：一维数组形式，表示样本标签。如果不是`{-1,1}`或者`{0,1}`的格式，那么参数`pos_label`需要显式设定
* `y_score`：一维数组形式，表示目标成绩。可以是对正样本的概率估计/置信度
* `pos_label`：指明正样本所属标签。如果`y_true`是`{-1,1}`或`{0,1}`格式，那么`pos_label`默认为`1`

```
import numpy as np
from sklearn.metrics import roc_curve

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)

fig = plt.figure()
plt.plot(fpr, tpr, label='ROC')
plt.show()
// 输出
[0.  0.  0.5 0.5 1. ]               # FPR
[0.  0.5 0.5 1.  1. ]               # TPR
[1.8  0.8  0.4  0.35 0.1 ]    # thresholds
```

返回的是`FPR、TPR`和阈值数组，`FPR`和`TPR`中每个坐标的值表示利用`thresholds`数组同样下标的阈值所得到的真阳性率和假阳性率

![](/imgs/ROC-AUC/roc_curve.png)

### roc_auc_score/auc

```
def roc_auc_score(y_true, y_score, average="macro", sample_weight=None,
                  max_fpr=None):
```

* `y_true`：格式为`[n_samples]`或者`[n_samples, n_classes]`
* `y_score`：格式为`[n_samples]`或者`[n_samples, n_classes]`

返回的是`AUC`的值

```
def auc(x, y, reorder='deprecated'):
```

* `x：FPR`
* `y：TPR`

利用`roc_curve`计算得到`FPR`和`TPR`后，就可以输入到`auc`计算`AUC`大小

```
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

print(roc_auc_score(y, scores))
print(auc(fpr, tpr))
# 输出
0.75
0.75
```