---
title: AdaGrad、RMSProp和Adam
abbrlink: 2bdd8f16
date: 2019-07-06 15:05:22
categories:
  - [算法, 最优化]
tags:
  - 逐元素自适应学习率方法
---

参考：

[Per-parameter adaptive learning rate methods](http://cs231n.github.io/neural-networks-3/#ada)

[lecture_slides_lec6.pdf](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

`AdaGrad、RMSProp`以及`Adam`都是逐元素的自适应学习率方法（`per-parameter adaptive learning rate methods`），根据每个神经元的梯度变化进行权重调整，能够有效的提高模型精度

## AdaGrad

数学公式如下：

$$
cache = cache + (dw)^{2}\\
w += -1 * lr / (\sqrt{cache} + eps)
$$

其中$w$是权重，$dw$是梯度，$lr$是学习率，使用$cache$累加梯度平方和

*$eps$是常量，通常设为`1e-8`，用于保证数值稳定性*

实现如下：

```
cache += dw**2
w += - learning_rate * dw / (np.sqrt(cache) + eps)
```

与原始$SGD$实现相比，有两点优势：

1. 其学习率除以$cache$的平方根，起到了学习率退火的效果
2. 如果得到了高梯度，则有效学习率下降，反之有效学习率提高，这样保证权重向量的变化更加稳定，不易被个别样本影响

其缺点在于变量$cache$是单调递增的，这导致学习率的单调递减，最终趋向于$0$，过早的停止学习

## RMSProp

数学公式如下：

$$
cache = decay_{rate} * cache + (1 - decay_{rate}) * (dw)^{2}\\
w += -1 * lr / (\sqrt{cache} + eps)
$$

其中$w$是权重，$dw$是梯度，$lr$是学习率，$decay_{rate}$表示衰减率，通常设为`[0.9, 0.99, 0.999]`其中之一，使用$cache$累加梯度平方和

*$eps$是常量，通常设为`1e-8`，用于保证数值稳定性*

实现如下：

```
cache = decay_rate * cache + (1 - decay_rate) * dw**2
w += - learning_rate * dw / (np.sqrt(cache) + eps)
```

与`AdaGrad`相比，其$cache$取值进行了弱化调整，通过[指数移动平均值](https://baike.baidu.com/item/EMA/12646151)的方式，避免梯度平方和（二阶动量）的单调累积，根据梯度变化进行自主调整，有效延长学习过程

## Adam

[Adam](https://arxiv.org/abs/1412.6980)方法集成了前两者，数学实现如下：

$$
m_{t} = \beta_{1}\cdot m_{t-1} + (1-\beta_{1})\cdot dw_{t}\\
v_{t} = \beta_{2}\cdot v_{t-1} + (1-\beta_{2})\cdot dw_{t}^{2}\\
\tilde{m_{t}} = m_{t} / (1 - \beta_{1})\\
\tilde{v_{t}} = v_{t} / (1 - \beta_{2})\\
w_{t} = w_{t-1} - lr \cdot \tilde{m_{t}} / (\sqrt{\tilde{v_{t}}} + \xi)
$$

$t$表示迭代次数

$\beta_{1}$和$\beta_{2}$是常量，取值在`[0,1]`之间

$lr$是学习率

$\xi$是常量，用于数值稳定，保证不除以$0$，取值在`[1e-4, 1e-8]`之间

常用的取值组合为$lr=0.001, \beta_{1}=0.9, \beta_{2}=0.999, \xi=10^{-8}$

参考：

![](/imgs/AdaGrad、RMSProp和Adam/adam_alg1.png)

实现如下：

```
m = beta1*m + (1-beta1)*dw
mt = m / (1-beta1**t)
v = beta2*v + (1-beta2)*(dw**2)
vt = v / (1-beta2**t)
w += - learning_rate * mt / (np.sqrt(vt) + eps)
```

`Adam`方法计算了梯度的一阶动量（均值，`mean`）和二阶动量（方差，`the uncentered variance`），同时为了避免初始动量不趋向于$0$，进行了偏置校正（`bias correction`）

与`RMSProp`方法相比，`Adam`方法进一步平滑了权重更新过程

## 梯度下降流程

文章[从 SGD 到 Adam —— 深度学习优化算法概览(一)](https://zhuanlan.zhihu.com/p/32626442)总结了梯度下降方法的更新框架

1. 计算梯度$dw$
2. 计算梯度的一阶动量$m_{t}$和二阶动量$v_{t}$
3. 更新权重$w_{t} = w_{t-1} - m_{t} / (\sqrt{v_{t} + \xi})$

## 小结

逐元素的自适应学习率方法能够更加有效的利用神经元的梯度变化，加速学习过程