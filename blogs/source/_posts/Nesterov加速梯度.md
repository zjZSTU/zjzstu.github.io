---
title: Nesterov加速梯度
categories:
  - [最优化]
  - [编程]
tags:
  - 动量更新
  - python
abbrlink: e51acd5
date: 2019-05-31 14:29:45
---

`Nesterov`加速梯度（`Nesterov's Accelerated Gradient`，简称`NAG`）是梯度下降的一种优化方法，其收敛速度比动量更新方法更快，收敛曲线更加稳定

## 实现公式

`NAG`计算公式参考[On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)

$$
w_{t-1}^{ahead} =w_{t-1} + \mu v_{t-1}\\ 
v_{t} = \mu v_{t-1} - lr \triangledown f(w_{t-1}^{ahead})\\
w_{t} = w_{t-1} + v_{t}
$$

其实现和经典动量的区别在于 **NAG计算的是当前权重加上累积速度后的梯度**

![](/imgs/Nesterov加速梯度/cm_nag.png)

### 替换公式

实际使用过程中使用替换公式，参考[Neural Networks Part 3: Learning and Evaluation ](http://cs231n.github.io/neural-networks-3/#sgd)

$$
w_{t} = w_{t-1} + v_{t}\\
\Rightarrow w_{t} +\mu v_{t} + \mu v_{t-1}= w_{t-1} + v_{t} + \mu v_{t} + \mu v_{t-1}\\
\Rightarrow w_{t}^{ahead} + \mu v_{t-1} = w_{t-1}^{ahead} + (1+\mu) v_{t}\\
\Rightarrow w_{t}^{ahead} = w_{t-1}^{ahead} + (1+\mu) v_{t} - \mu v_{t-1}\\
$$

所以替换公式为

$$
v_{t} = \mu v_{t-1} - lr \triangledown f(w_{t-1}^{ahead})\\
w_{t}^{ahead} = w_{t-1}^{ahead} + (1+\mu) v_{t} - \mu v_{t-1}
$$

$$
\Rightarrow
$$

$$
v_{t} = \mu v_{t-1} - lr \triangledown f(w_{t-1})\\
w_{t} = w_{t-1} + (1+\mu) v_{t} - \mu v_{t-1}
$$

将使用的权重向量替换为权重向量加上累积速度后的权重值

## numpy测试

参考：[动量更新](https://www.zhujian.tech/posts/2b34c959.html#more)

### 原始公式实现

`NAG`实现代码如下

```
def sgd_nesterov(x_start, lr, epochs=100, mu=0.5):
    dots = [x_start]

    x = x_start.copy()
    v = np.zeros((2))
    for i in range(epochs):
        x_ahead = x + mu * v
        grad = gradient(x_ahead)
        v = mu * v - lr * grad
        x += v
        dots.append(x.copy())
        if abs(np.sum(grad)) < 1e-6:
            break
    return np.array(dots)
```

#### SGD vs CM vs NAG

学习率`1e-3`，动量因子`0.5`，迭代`100`次结果

![](/imgs/Nesterov加速梯度/sgd_cm_nag_1.png)

学习率`1e-2`，动量因子`0.5`，迭代`100`次结果

![](/imgs/Nesterov加速梯度/sgd_cm_nag_2.png)

**经典动量和NAG方法的收敛速度均比标准梯度下降更快**

#### CM vs NAG

学习率`1e-3`，动量因子`0.9`，迭代`100`次结果

![](/imgs/Nesterov加速梯度/cm_nag_1.png)

学习率`1e-2`，动量因子`0.9`，迭代`100`次结果

![](/imgs/Nesterov加速梯度/cm_nag_2.png)

**NAG方法的收敛曲线比经典动量方法更稳定**

### 替换公式实现

替换公式实现代码如下：

```
def sgd_nesterov_v2(x_start, lr, epochs=100, mu=0.5):
    dots = [x_start]

    x = x_start.copy()
    v = 0
    for i in range(epochs):
        grad = gradient(x)
        v_prev = v
        v = mu * v - lr * grad
        x += (1 + mu) * v - mu * v_prev
        dots.append(x.copy())
        if abs(np.sum(grad)) < 1e-6:
            break
    return np.array(dots)
```

#### NAG vs NAG_alter

学习率`1e-3`，动量因子`0.9`，迭代`100`次结果

![](/imgs/Nesterov加速梯度/nag_alter_1.png)

学习率`1e-2`，动量因子`0.9`，迭代`100`次结果

![](/imgs/Nesterov加速梯度/nag_alter_2.png)

学习率`1e-2`，动量因子`0.5`，迭代`100`次结果

![](/imgs/Nesterov加速梯度/nag_alter_3.png)

## 相关资料

[Nesterov Accelerated Gradient and Momentum](https://jlmelville.github.io/mize/nesterov.html)

[比Momentum更快：揭开Nesterov Accelerated Gradient的真面目](https://zhuanlan.zhihu.com/p/22810533)