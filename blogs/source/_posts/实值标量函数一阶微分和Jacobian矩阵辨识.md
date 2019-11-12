---
title: 实值标量函数一阶微分和Jacobian矩阵辨识
categories: 数学
tags:
  - [微积分]
  - [线性代数]
abbrlink: b9ab243b
date: 2019-05-14 14:23:29
---

参考：《矩阵分析与应用》第三章 3.2 一阶实矩阵微分与Jacobian矩阵辨识

神经网络的反向传播可以通过对损失函数进行微分得到各层权重矩阵的梯度

其中对损失函数求梯度是实值标量函数一阶微分，其中关键的部分是得到Jacobian矩阵，从而转置获取梯度矩阵

## 一阶实矩阵微分

矩阵微分用符号$dX$表示，定义为$dX=[dX_{ij}]_{i=1,j=1}^{m,n}$

实矩阵微分具有两个基本性质

1. 转置。矩阵转置的微分等于矩阵微分的转置，既有$d(X^{T})=(dX)^{T}$
2. 线性。$d(\alpha X+\beta Y)=\alpha dX+\beta dY$

### 常用计算公式

1. 常数矩阵的微分矩阵为零矩阵，即$dA=O$
2. 常数$\alpha$与矩阵$X$的乘积的微分矩阵$d(\alpha X)=\alpha dX$
3. 矩阵转置的微分矩阵等于原矩阵的微分矩阵的转置，即$d(X^{T})=(dX)^{T}$
4. 两个矩阵函数的和（差）的微分矩阵为$d(U\pm V)=dU\pm dV$
5. 常数矩阵与矩阵乘积的微分矩阵为$d(AXB)=A(dX)B$
6. 矩阵函数$U=F(X),V=G(X),W=H(X)$乘积的微分矩阵为
    $$
    d(UV)=(dU)V+U(dV)\\
    d(UVW)=(dU)VW+U(dV)W+UV(dW)
    $$
7. 矩阵$X$的迹的矩阵微分$d(tr(X))$等于矩阵微分$dX$的迹$tr(dX)$，即
    $$
    d(tr(X))=tr(dX)
    $$
    7.1 从而可推导出矩阵函数$F(X)$的迹的矩阵微分为$d(tr(F(X)))=tr(d(F(X)))$
8. 行列式的微分为
    $$
    d|X|=|X|tr(X^{-1}dX)
    $$
    8.1 从而可推导出矩阵函数$F(X)$的行列式的微分为$d(|F(X)|)=|F(X)|tr(F^{-1}(X)d(F(X)))$
9. 矩阵函数的`Kronecker`积的微分矩阵为
    $$
    d(U\bigotimes V)=(dU)\bigotimes V+U\bigotimes dV
    $$
10. 矩阵函数的`Hadamard`积的微分矩阵为
    $$
    d(U* V)=(dU)* V+U* dV
    $$
11. 向量化函数$vec(X)$的微分矩阵等于$X$的微分矩阵的向量化函数，即
    $$
    d(vec(X))=vec(dX)
    $$
12. 矩阵对数的微分矩阵为
    $$
    d\log X=X^{-1}dX
    $$
    12.1 从而可推导出矩阵函数$F(X)$的对数的微分矩阵为$d(\log F(X))=F^{-1}(X)d(F(X))$
13. 逆矩阵的微分矩阵为
    $$
    d(X^{-1})=-X^{-1}(dX)X^{-1}
    $$

## 标量函数的Jacobian矩阵辨识

多变量函数$f(x_{1},...,x_{m})$在点$(x_{1},...,x_{m})$可微分的充分条件是偏导数$\frac {\partial f}{\partial x_{1}},...,\frac {\partial f}{\partial x_{m}}$均存在，且连续。全微分公式如下：

$$
df(x_{1},...,x_{m})=\frac {\partial f}{\partial x_{1}}dx_{1}+...+\frac {\partial f}{\partial x_{m}}dx_{m}
$$

若矩阵的标量函数$f(x)$在$m\times n$矩阵点$X$可微分，则$Jacobian$矩阵可直接通过以下公式辨识：

$$
df(x)=tr(Adx)\Leftrightarrow D_{x}f(x)=A\\
df(X)=tr(AdX)\Leftrightarrow D_{X}f(X)=A
$$

要点如下：

1. 标量函数$f(X)$总可以写成迹函数的形式，因为$f(X)=tr(f(X))$
2. 无论$dX$出现在迹函数内的任何位置，总可以通过迹函数的性质$tr[A(dX)B]=tr(BAdX)$，将$dX$写到迹函数的最右端，从而得到迹函数微分矩阵的规范形式
3. 对于$(dX)^{T}$，总可以通过迹函数的性质$tr[A(dX)^{T}B]=tr(A^{T}B^{T}dX)$，写成迹函数微分矩阵的规范形式

计算标量函数$f(x)=x^{T}Ax$，$A$是正方常数矩阵，求梯度矩阵

$$
df(x)=d(tr(x^{T}Ax))
=tr[(dx)^{T}Ax+x^{T}Adx]\\
=tr([(dx)^{T}Ax]^{T}+x^{T}Adx)
=tr(x^{T}A^{T}dx+x^{T}Adx)
=tr(x^{T}(A+A^{T})dx)
$$

所以$Jacobian$矩阵为$x^{T}(A+A^{T})$，梯度矩阵为$(A+A^{T})x$

计算$tr(X^{T}X)$的梯度矩阵

$$
dtr(X^{T}X)=tr(d[X^{T}X])=tr((dX)^{T}X+X^{T}dX)\\
=tr((dX)^{T}X)+tr(X^{T}dX)
=tr(X^{T}dX)+tr(X^{T}dX)
=tr(2X^{T}dX)
$$

所以$Jacobian$矩阵为$2X^{T}$，梯度矩阵为$2X$

常用的迹函数的微分矩阵及其$Jacobian$矩阵参考`《矩阵分析与应用》第3.2章表3.2.1`