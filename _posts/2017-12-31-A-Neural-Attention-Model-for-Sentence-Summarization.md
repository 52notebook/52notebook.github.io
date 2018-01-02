---
layout: post
title: A Neural Attention Model for Sentence Summarization
description: "论文利用`local attention mechanism`基于`input sentence`建模生成摘要的每一个词。"
modified: 2017-12-31T17:00:45-04:00
tags: [ANN, RNN, 梯度, 梯度爆炸]
---


- 思想
    - 论文利用`local attention mechanism`基于`input sentence`建模生成摘要的每一个词。
- 该摘要类型: `Abstractive `而非`extractive`。
- 模型score函数

$$s(x, y) \approx \sum_{i=0}^{N-1}g(y_{i+1}, x, y_c)$$

`$N$`是Output length，论文假定是固定的，且预先设定的。
`y_c`是大小的`$C$`的词窗口，计算如下:

$$y_c = [y_{i-C+1}, ...y_{i}]$$

`$g$`函数常用`conditional log probability`，因此`$s(x,y)$`可用以下表示

$$s(X, Y) = log(Y|X; \theta) \approx \sum_{i=0}^{N-1}log(Y_{i+1}, X, y_c)$$
- 语言模型

`$log(Y|X; \theta)$`是条件语言模型，核心任务是计算下一个词的概率分布。
- NNLM

经典NNLM模型如下图

![image](http://note.youdao.com/yws/public/resource/645c7ef0f51ed836661b0eb73a4e7366/xmlnote/63BDBE23B37E4B5584CE40FE99805FFA/1849)

Beigio提出的经典神经网络语言模型如下:


$$p(y_{i+1}|y_c,X;\theta) \quad \infty \quad exp(Vh + W_{enc(X,y_c)})$$

$$\hat y_c = [Ey_{i-C+1},....Ey_{i}]$$

$$h = tanh(U \hat y_c)$$




`$E$`是词嵌套矩阵

`$U,V,W$`是权重矩阵

`$h$`是隐藏层网络

`$enc$`是上下文编码器，对`input`和当前`context`的表征。

- 论文讨论的重点是`Encoder`使用上，分别介绍了三种`Encoder`网络结构:
    - Bag of Words Encoder
    - Convolutional Encoder
    - Attention Encoder

本文提出的基于Attention mechanism的Encoder网络结构如下图


![image](http://note.youdao.com/yws/public/resource/645c7ef0f51ed836661b0eb73a4e7366/xmlnote/756D722136924F7CBE044CD21E19616C/1856)


- Attention Encoder



$$enc(X,y_c) = p^{T}x^{-}$$

$$p \quad \infty \quad exp(\hat x P \hat y_c')$$

$$\hat x = [Fx_1,...,Fx_M]$$

$$y_c' = [G_{y_{i-C+1}},...,G_{y_i}]$$

$$x^{-}_{i} = \sum_{q=i-Q}^{i+Q} \hat x_i / Q$$

`$P$`是`input embedding matrix`与`context matrix`间映射形成的新的权重矩阵

`$F$`是`$word \quad embedding$`矩阵

`$G$`是`$context \quad embedding$`


该算法取得了不错准确率效果，生成的摘要句子语法有待进一步改善。
