---
layout: post
title:  introduction to conditional random fields
description: "以POS词性标注任务阐述CRF，从模型训练到推理逐层深入，读完帮助理解CRF可以做什么以及是怎么做的。CRF是最大熵在sequence上的扩展，也是HMM conditional的求解。"
modified: 2018-02-14T17:00:45-04:00
tags: [CRF, POS, 词性标注]
---



### 1 背景知识

#### 1.1 HMM

**HMM(隐马尔科夫)**是用来对**序列数据X做标注Y的生成模型，用马尔科夫链(Markov Chain)对联合概率$P(X,Y)$建模**:
$$
P(X,Y)= \Pi_{t} P(y_t|y_{t-1})P(x_t|y_t)
$$
然后通过**Viterbi算法**求解P(Y|X)的最大值。
<!-- more -->




* [1 背景知识](#1-背景知识)
   * [1.1 HMM](#11-hmm)
   * [1.2 LR逻辑回归](#12-lr逻辑回归)
* [2 CRF](#2-crf)
* [3 基本问题](#3-基本问题)
* [4 训练过程](#4-训练过程)
* [5 推理](#5-推理)
* [6 CRF应用于词性标注任务例子](#6-crf应用于词性标注任务例子)
   * [6.1 特征函数](#61-特征函数)
   * [6.2 估算概率](#62-估算概率)
      * [6.2.1 特征函数](#621-特征函数)
* [7 权重学习](#7-权重学习)
* [8 查找最佳标签(状态)](#8-查找最佳标签状态)
* [9 模型比较](#9-模型比较)
   * [9.1 与LR](#91-与lr)
   * [9.2 与HMM](#92-与hmm)
      * [9.2.1 比HMM高明处](#921-比hmm高明处)



### 1 背景知识

#### 1.1 HMM

**HMM(隐马尔科夫)**是用来对**序列数据X做标注Y的生成模型，用马尔科夫链(Markov Chain)对联合概率$P(X,Y)$建模**:
$$
P(X,Y)= \Pi_{t} P(y_t|y_{t-1})P(x_t|y_t)
$$
然后通过**Viterbi算法**求解P(Y|X)的最大值。

#### 1.2 LR逻辑回归



LR模型是分类任务中的判别模型，直接使用Ligistic函数**建模条件概率P(y|x)**。实际上，logistic函数是softmax的特殊形式，并且**LR等价于最大熵模型**，完全可以写成最大熵的形式：

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1ihg6lw20j207g02ba9x.jpg)

其中，Zw(x)为归一化因子，w为模型的参数，fi(x,y)为特征函数（feature function）——描述(x,y)的某一事实。

### 2 CRF

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1qnfpltowj20de03jt8u.jpg)

CRF是为了解决标注问题的判别模型。**CRF是最大熵模型的sequence扩展、HMM的conditional 求解**

CRF考虑的是序列的整体，如果单独从整体中取个体，这种会损失很多信息。比如，给定一段小视频，要为其内容每个动作打标签，张口这个个体既可能是吃饭，也可能会是唱歌，这就是忽略了序列整体的例子。因此，为了提高性能，需要考虑附近帧(活动的)标签信息，这就是CRF做到事情。

[Introduction to Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1ihfh8n4bj20cy09d74u.jpg)



CRF是计算给定观察序列输入$X=(x_1,....x_n) \in X^N$，输出状态序列条件概率$p(Y|X)$,其中$Y=(y_1,....y_n) \in y^N$。

### 3 基本问题

线性CRF可以以HMM相似方式表示成以下基本问题:

- 给定观察集和CRF模型，如何找到最大程度拟合状态序列?
- 给定状态序列和观察序列，如何找到参数使得条件概率值最大?

问题1是CRF最常用的使用场景，问题2是关于如何训练和调整参数的。

**在判别模型中，p(X|Y)概率不需要考虑，那是HMM(生成模型)常见解决问题，CRF并不需要考虑这点**

### 4 训练过程

各种类型的CRF都可以称为是最大熵模型，是一种参数估计，也就是说通过最大对数似然损失函数进行训练。

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1qdj0sp5ej20js04hwet.jpg)

添加惩罚项后的Loss Function

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1qdmt5tf1j20kh0cygn1.jpg)

记住，训练过程是使用**Forward-Backward**算法更新参数

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1qnfhtbpej20kr0c8ab0.jpg)

### 5 推理

CRF推理任务是指给定观察序列X下，找到最可能的状态序列Y。**维特比算法和Forward-Backward算法相似，主要不同是，维特比使用最大值替代求和运算**，我也不知道什么意思，具体看下维特比计算任务:

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1qf40tw2ej20ki03tq34.jpg)

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1qf4ees45j20kf0l2tax.jpg)

### 6 CRF应用于词性标注任务例子

在词性标注任务中，给定一段文本序列为其每个词打上标签，如ADJECTIVE, NOUN, PREPOSITION, VERB, ADVERB, ARTICLE.

#### 6.1 特征函数

构建CRF，首先要设置一组特征函数，特征函数的目的是评估给定前一个观察序列下，当前观察序列被标记成各状态概率大小。

通常，特征函数的输入包括以下:

- 观察序列 S
- 观察序列当前词的索引
- 当前词的状态(标签)
- 前一个词的状态(标签)

而且一般输出值是实数值，通常是1或0.

比如，设置一个特征函数用来评价给定前一个词是"very"下，当前词的状态被标记成`Adj`的概率。

#### 6.2 估算概率

为每个特征函数$f_j$分配一个$w_j$,权重是通过训练学习而来。这样以来，给定一条观察序列 $S$,标签$l$得分是序列中所有词的特征函数加权得分。

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1qu8s1uoej20it02x3yk.jpg)

然后，把该得分通过exp和归一化运算转成概率值。

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1quclbisuj20c301mdfr.jpg)

##### 6.2.1 特征函数

以POS标注任务为例，一般特征函数包括如下:

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1quef5ekcj20iv0930uc.jpg)

总结起来，CRF构建过程概括

- 首先，要定义一批的特征函数，如上面那种样式。
- 然后，为每个特征函数赋个权重，标明该特征函数的重要性。
- 接着，把以上很多的特征函数为观察序列打的分值相加。
- 最后，就是将加权得分值转成概率值。

### 7 权重学习

如何更新CRF特征函数的权重呢，答案用梯度法呗。具体见下:

![](https://ws1.sinaimg.cn/large/e318ef1dly1g1rusuib3vj20j105pjsg.jpg)

### 8 查找最佳标签(状态)

要考虑的问题是，加入有一个已经训练好的CRF模型，对应一个新的序列，如何找到最优的标签(状态)？

最简单容易想到的方法是，对每个可能 标签(状态)计算$P(l|s)$ 概率，然后再选取一个能够使得概率取得最大的标签。然而这种方法计算时间复杂度是$k^m$，计算量随序列长度增大指数级增大(可能标签有K种，序列长度是m)。

有一种更好的解决方式是利用CRF具有最佳子路径的性质，使用**动态规划**算法寻找最优标签(状态)，正如HMM中的**维特比**算法做得一样。


### 9 模型比较

#### 9.1 与LR

CRF看起来很熟悉，想LR逻辑回归，对就是的，只不过是LR是对数线性常用来做分类任务，而CRF也是对数线性的却是干序列标注的。可以说是CRF是LR在序列上的扩展。

#### 9.2 与HMM

HMM是什么，他也可以做序列标注，比如POS词性标注任务等。不同之处在于CRF是用一组特征函数计算标签得分，而HMM是采样一种生成方式先标注后计算条件概率，公式如下:

![](https://ws1.sinaimg.cn/large/e318ef1dgy1g1quzj268fj20hw03zt8x.jpg)

CRF要比HMM强大的多，HMM能建模的CRF均可而且HMM不能建模的CRF仍可以。

HMM对数概率 $$log P(l,s) = log P(l_0 ) + \sum_i log P(l_i|l_{i-1}) + \sum_I log P(w_i|l_i)$$

 上面公式，如果我们把对数概率看成是附带权重的转移和发射指示特征函数时，那么就是CRF对数线上形式了。

总之，我们能够通过下面方式把CRF转成HMM，主要是根据转移矩阵和发射矩阵改成特征函数。方式如下:

- 对于HMM的转移概率 $P(l_i=y|l_{i-1} = x)$，要为CRF定义如下样子的特征函数$f_{xy}( s, i, l_i, l_{i-1}) = 1$ 如果$l_i =y   $且$l_{i-1} = x$ ,并且为每个特征函数赋如下的权重 $w_{xy} = log P(l_i = y|l_{i_1} = x)$

- 对于HMM发射概率矩阵$P(w_i=z|l_i=x)$，对应定义CRF的"发射特殊函数"，形如$g_{xy}(s,i,l_i,l_{i-1}) = 1$， 如果$w_i=z$ 并且$l_i = x$。与之对应的每个特征函数的权重$w_{x,z} = log P(w_i=z|l_i=x)$

##### 9.2.1 比HMM高明处

- CRF能够定义更多的特征函数

    HMM带有天然的局部属性，比如它受转移矩阵和发射矩阵的约束，**强制约束每个词只能依赖当前标签，并且当前标签只能依赖前一个标签(状态)；CRF能够利用全局标签**。

- CRF能够为特征函数赋予任意大小的权重值。而HMM的必须满足统计概率约束，即$\sum_wP(w_i=w|l_1)=1$

[Ref](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
[Ref](https://gist.github.com/wut0n9/e6132d1d4195d08410e2aa303cf5a3f4)