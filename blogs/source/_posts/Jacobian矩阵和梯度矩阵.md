---
title: Jacobian矩阵和梯度矩阵
abbrlink: '29422005'
date: 2019-05-13 11:01:53
categories: 数学
tags:
  - 微积分
  - 线性代数
---

参考：《矩阵分析与应用》第3章 3.1 Jacobian矩阵与梯度矩阵

在`pytorch`的`autograd`包中，利用`Jacobian`（雅格比）矩阵进行梯度的计算

学习实值标量函数、实值向量函数和实值矩阵函数相对于实向量变元或矩阵变元的偏导

## 计算符号

* 实向量变元：$x=[x_{1},...,x_{m}]^T\in R^{m}$
* 实矩阵变元：$X=[x_{1},...,x_{n}]\in R^{m\times n}$
* 实值标量函数
    * $f(X)\in R$，其变元是$m\times 1$实值向量$x$，记作$f:R^{m}\rightarrow R$
    * $f(X)\in R$，其变元是$m\times n$实矩阵$X$，记作$f:R^{m\times n}\rightarrow R$
* $p$维实列向量函数
    * $f(x)\in R^{p}$，其变元是$m\times 1$实值向量$x$，记作$f:R^{m}\rightarrow R^{p}$
    * $f(X)\in R^{p}$，其变元是$m\times n$实矩阵$X$，记作$f:R^{m}\rightarrow R^{p}$
* $p\times q$维实矩阵函数
    * $f(x)\in R^{p\times q}$，其变元是$m\times 1$实值向量$x$，记作$f:R^{m}\rightarrow R^{p\times q}$
    * $f(X)\in R^{p\times q}$，其变元是$m\times n$实矩阵$X$，记作$f:R^{m}\rightarrow R^{p\times q}$

实值函数的分类

| 函数类型 	| 向量变元$x\in R^{m}$	| 矩阵变元$X\in R^{m\times n}$ 	|
|:--------:	|:--------:	|------	|
| 标量函数$f\in R$ | $f(x), \ f: R^{m}\rightarrow R$ |  $f(X), \ f: R^{m\times n}\rightarrow R$|
| 向量函数$f\in R^{p}$ | $f(x), \ f: R^{m}\rightarrow R^{p}$ |  $f(X), \ f: R^{m\times n}\rightarrow R^{p}$|
| 矩阵函数$F\in R^{p\times q}$ | $F(x), \ f: R^{m}\rightarrow R^{p\times q}$ |  $F(X), \ F: R^{m\times n}\rightarrow R^{p\times q}$|

## 行向量偏导算子和Jacobian矩阵

### 实值标量函数

定义实向量变元$x=[x_{1},...,x_{m}]^T$，$1\times m$行向量偏导算子记为

$$
D_{x}=\frac {\partial }{\partial x^T}
=[\frac {\partial }{\partial x_{1}},...,\frac {\partial }{\partial x_{m}}]
$$

对于实值标量函数$f(x)$而言，对于$x$的偏导向量是一个$1\times m$行向量

$$
D_{x}f(x)=\frac {\partial f(x)}{\partial x^T}
=[\frac {\partial f(x)}{\partial x_{1}},...,\frac {\partial f(x)}{\partial x_{m}}]
$$

当变元为实值矩阵$X\in R^{m\times n}$时，其偏导向量有两种表示形式

$$
D_{X}f(X)=\frac {\partial f(X)}{\partial X^T}=
\begin{bmatrix}
\frac {\partial f(X)}{\partial x_{11}} & \dots & \frac {\partial f(X)}{\partial x_{m1}}\\ 
\vdots & \vdots & \vdots\\ 
\frac {\partial f(X)}{\partial x_{1n}} & \vdots & \frac {\partial f(X)}{\partial x_{mn}}
\end{bmatrix}
\in R^{n\times m}
$$

或者

$$
D_{vecX}f(X)=[\frac {\partial f(X)}{\partial x_{11}},...,\frac {\partial f(X)}{\partial x_{m1}},...,\frac {\partial f(X)}{\partial x_{1n}},...,\frac {\partial f(X)}{\partial x_{mn}}]
$$

$D_{X}f(X)$称为实值标量函数$f(X)$关于矩阵变元$X$的$Jacobian$矩阵

$D_{vecX}f(X)$称为实值标量函数$f(X)$关于矩阵变元$X$的**行偏导向量**

两者之间关系

$$
D_{vecX}f(X)=rvec(D_{X}f(X))=(vec(D_{X}^{T}f(X)))^T
$$

即实值标量函数$f(X)$的行向量偏导$D_{vecX}f(X)$等于$Jacobian$矩阵的转置$D_{X}^{T}f(X)$的列向量化$vec(D_{X}^{T}f(X)$的转置

### 实值矩阵函数

计算实值矩阵函数$F(X)=[f_{kl}]_{k=1,l=1}^{p,q}\in R^{p\times q}$对于矩阵变元$X\in R^{m\times n}$的行偏导矩阵:

先通过列向量化，将$p\times q$矩阵函数$F(X)$转换成$pq\times 1$列向量

$$
vec(F(X))=
[f_{11}(X),...,f_{p1}(X),...,f_{1q}(X),...,f_{pq}(X)]^T\in R^{pq}
$$

然后，将该列向量对变元$X$的列向量化的转置$(vecX)^T$求偏导，给出$pq\times mn$维$Jacobian$矩阵

$$
D_{X}F(X)=\frac {\partial vec(F(X))}{\partial (vecX)^T}\in R^{pq\times mn}
$$

具体表达式如下：

$$
D_{X}F(X)=
\begin{bmatrix}
\frac {\partial f_{11}}{\partial (vecX)^T}\\ 
\vdots\\ 
\frac {\partial f_{p1}}{\partial (vecX)^T}\\ 
\vdots\\ 
\frac {\partial f_{1q}}{\partial (vecX)^T}\\ 
\vdots\\ 
\frac {\partial f_{pq}}{\partial (vecX)^T}
\end{bmatrix}=
\begin{bmatrix}
\frac {\partial f_{11}}{\partial x_{11}} & \dots && \frac {\partial f_{11}}{\partial x_{m1}} & \dots & \frac {\partial f_{11}}{\partial x_{1n}} & \dots & \frac {\partial f_{11}}{\partial x_{mn}}\\ 
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\ 
\frac {\partial f_{p1}}{\partial x_{11}} & \dots && \frac {\partial f_{p1}}{\partial x_{m1}} & \dots & \frac {\partial f_{p1}}{\partial x_{1n}} & \dots & \frac {\partial f_{p1}}{\partial x_{mn}}\\ 
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
\frac {\partial f_{1q}}{\partial x_{11}} & \dots && \frac {\partial f_{1q}}{\partial x_{m1}} & \dots & \frac {\partial f_{1q}}{\partial x_{1n}} & \dots & \frac {\partial f_{1q}}{\partial x_{mn}}\\  
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
\frac {\partial f_{pq}}{\partial x_{11}} & \dots && \frac {\partial f_{pq}}{\partial x_{m1}} & \dots & \frac {\partial f_{pq}}{\partial x_{1n}} & \dots & \frac {\partial f_{pq}}{\partial x_{mn}}\\ 
\end{bmatrix}
$$

## 列向量偏导算子和梯度矩阵

**采用列向量形式定义的偏导算子称为列向量偏导算子，又称为梯度算子**

### 实值标量函数

定义实向量变元$x=[x_{1},...,x_{m}]^T$，$1\times m$行向量偏导算子记为

$$
\bigtriangledown_{x}=\frac {\partial }{\partial x^T}
=[\frac {\partial }{\partial x_{1}},...,\frac {\partial }{\partial x_{m}}]^T
$$

对于实值标量函数$f(x)$而言，对于$x$的偏导向量$\bigtriangledown_{x}f(x)$是一个$m\times 1$列向量

$$
D_{x}f(x)=\frac {\partial f(x)}{\partial x}
=[\frac {\partial f(x)}{\partial x_{1}},...,\frac {\partial f(x)}{\partial x_{m}}]^T
$$

将实值矩阵变元$X\in R^{m\times n}$列向量化后，关于矩阵变元$X$的梯度向量为

$$
\bigtriangledown_{vecX}f(X)=\frac {\partial f(X)}{\partial vecX}
=[\frac {\partial f(X)}{\partial x_{11}},...,\frac {\partial f(X)}{\partial x_{m1}},...,\frac {\partial f(X)}{\partial x_{1n}},...,\frac {\partial f(X)}{\partial x_{mn}}]^T
$$

或者

$$
\bigtriangledown_{X}f(X)=\frac {\partial f(X)}{\partial X}=
\begin{bmatrix}
\frac {\partial f(X)}{\partial x_{11}} & \dots & \frac {\partial f(X)}{\partial x_{1n}}\\ 
\vdots & \vdots & \vdots\\ 
\frac {\partial f(X)}{\partial x_{m1}} & \vdots & \frac {\partial f(X)}{\partial x_{mn}}
\end{bmatrix}
$$

前者称为实值标量函数$f(X)$关于实值矩阵变元$X$的列向量偏导算子

后者称为实值标量函数$f(X)$关于实值矩阵变元$X$的梯度矩阵

**所以实值标量函数$f(X)$的梯度矩阵等于$Jacobian$矩阵的转置**

$$
\bigtriangledown_{X}f(X)=D_{X}^T f(X)
$$

### 实值矩阵函数

计算实值矩阵函数$F(X)\in R^{p\times q}$对于矩阵变元$X\in R^{m\times n}$的梯度矩阵

先通过列向量化，将$p\times q$矩阵函数$F(X)$转换成$pq\times 1$列向量

$$
vec(F(X))=
[f_{11}(X),...,f_{p1}(X),...,f_{1q}(X),...,f_{pq}(X)]^T\in R^{pq}
$$

然后，将该列向量对变元$X$的列向量化$vecX$求偏导，给出$pq\times mn$维梯度矩阵

具体表达式如下：

$$
\bigtriangledown_{X}F(X)=
\begin{bmatrix}
\frac {\partial f_{11}}{\partial vecX}\\ 
\vdots\\ 
\frac {\partial f_{p1}}{\partial vecX}\\ 
\vdots\\ 
\frac {\partial f_{1q}}{\partial vecX}\\ 
\vdots\\ 
\frac {\partial f_{pq}}{\partial vecX}
\end{bmatrix}=
\begin{bmatrix}
\frac {\partial f_{11}}{\partial x_{11}} & \dots && \frac {\partial f_{11}}{\partial x_{11}} & \dots & \frac {\partial f_{11}}{\partial x_{11}} & \dots & \frac {\partial f_{11}}{\partial x_{11}}\\ 
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\ 
\frac {\partial f_{p1}}{\partial x_{m1}} & \dots && \frac {\partial f_{p1}}{\partial x_{m1}} & \dots & \frac {\partial f_{p1}}{\partial x_{m1}} & \dots & \frac {\partial f_{p1}}{\partial x_{m1}}\\ 
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
\frac {\partial f_{1q}}{\partial x_{1n}} & \dots && \frac {\partial f_{1q}}{\partial x_{1n}} & \dots & \frac {\partial f_{1q}}{\partial x_{1n}} & \dots & \frac {\partial f_{1q}}{\partial x_{1n}}\\  
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
\frac {\partial f_{pq}}{\partial x_{mn}} & \dots && \frac {\partial f_{pq}}{\partial x_{mn}} & \dots & \frac {\partial f_{pq}}{\partial x_{mn}} & \dots & \frac {\partial f_{pq}}{\partial x_{mn}}\\ 
\end{bmatrix}
$$

**所以实值矩阵函数$f(X)$的梯度矩阵等于$Jacobian$矩阵的转置**

$$
\bigtriangledown_{X}F(X)=(D_{X} F(X))^T
$$

## 偏导和梯度计算

实值函数对于矩阵变元$X$的梯度计算有如下性质和法则

1. 若$f(X)=c$为常数，其中$X\in R^{m\times n}$，则梯度$\frac {\partial c}{\partial X}=O_{m\times n}$（**维数相容原则**）
2. 线性法则。若$f(X)$和$g(X)$分别是矩阵$X$的实值函数，$c_{1}$和$c_{2}$为实常数，那么

$$
\frac {\partial [c_{1}f(X)+c_{2}g(X)]}{\partial X}
=c_{1}\frac {\partial f(X)}{\partial X}
+c_{2}\frac {\partial g(X)}{\partial X}
$$

3. 乘积法则。若$f(X), g(X)$和$h(X)$都是矩阵$X$的实值函数，则

$$
\frac {\partial [f(X)g(X)]}{\partial X}
=g(X)\frac {\partial f(X)}{\partial X}
+f(X)\frac {\partial g(X)}{\partial X}
$$

以及

$$
\frac {\partial [f(X)g(X)h(X)]}{\partial X}
=g(X)h(X)\frac {\partial f(X)}{\partial X}
+f(X)h(X)\frac {\partial g(X)}{\partial X}
+f(X)g(X)\frac {\partial h(X)}{\partial X}
$$

4. 商法则。若$g(X)\neq 0$，则

$$
\frac {\partial [f(X)/g(X)]}{\partial X}
=\frac {1}{g(X)^2}[g(X)\frac {\partial f(X)}{\partial X}-f(X)\frac {\partial g(X)}{\partial X}]
$$

5. 链式法则。令$X$为$m\times n$矩阵，且$y=f(X)$和$g(y)$分别是以矩阵$X$和标量$y$为变元的实值函数，则

$$
\frac {\partial g(f(X))}{\partial X}
=\frac {dg(y)}{dy} \frac {\partial f(X)}{\partial X}
$$

### 实值标量函数

针对实值标量函数有如下推论

1. 实值函数$f(x)=x^{T}Ax$的行偏导向量为$Df(x)=x^{T}(A+A^{T})$，梯度向量为$\bigtriangledown_{X}f(x)=(Df(X))^{T}=(A^{T}+A)x$
2. 实值函数$f(x)=a^{T}XX^{T}b$，其中$X\in R^{m\times n},a,b\in R^{n\times 1}$，$Jacobian$矩阵为$D_{X}f(X)=X^{T}(ba^{T}+ab^{T})$，梯度矩阵为$\bigtriangledown_{X}f(x)=(ab^{T}+ba^{T})X$
3. 实值函数$f(X)=tr(XB)$，其中$X\in R^{m\times n}, b\in R^{n\times m}, tr(BX)=tr(XB)$，所以$Jacobian$矩阵为$D_{X}tr(XB)=D_{X}tr(BX)=B$，梯度矩阵为$\bigtriangledown_{X}tr(XB)=\bigtriangledown_{X}tr(BX)=B^{T}$

以推论一为例，假设

$$
x = \begin{bmatrix}
x_{1}\\ 
x_{2}
\end{bmatrix} \ 
A=\begin{bmatrix}
a_{11} & a_{12}\\ 
a_{21} & a_{22}
\end{bmatrix}
$$

所以

$$
f(x)=x^{T}Ax=
\begin{bmatrix}
x_{1} & x_{2}
\end{bmatrix}
\begin{bmatrix}
a_{11} & a_{12}\\ 
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
x_{1}\\ 
x_{2}
\end{bmatrix}
=\sum_{k=1}^{2}\sum_{l=1}^{2}a_{kl}x_{k}x_{l}
$$

$$
=[x_{1}a_{11}+x_{2}a_{21}, x_{1}a_{12}+x_{2}a_{22}]
\begin{bmatrix}
x_{1}\\ 
x_{2}
\end{bmatrix}
=x_{1}a_{11}x_{1}+x_{2}a_{21}x_{1}+x_{1}a_{12}x_{2}+x_{2}a_{22}x_{2}
$$

$$
Df(X)=\frac {\partial f(x)}{\partial x}=
[x_{1}a_{11}+a_{11}x_{1}+x_{2}a_{21}+a_{12}x_{2}, a_{21}x_{1}+x_{1}a_{12}+x_{2}a_{22}+a_{22}x_{2}]=\\
[x_{1}a_{11}+x_{2}a_{21}, x_{1}a_{12}+x_{2}a_{22}]
+[a_{11}x_{1}+a_{12}x_{2}, a_{21}x_{1}+a_{22}x_{2}]\\
=[x_{1},x_{2}]\begin{bmatrix}
a_{11} & a_{12}\\ 
a_{21} & a_{22}
\end{bmatrix}
+[x_{1},x_{2}]\begin{bmatrix}
a_{11} & a_{21}\\ 
a_{12} & a_{22}
\end{bmatrix}
=x^{T}A+x^{T}A^{T}
=x^{T}(A+A^{T})
$$