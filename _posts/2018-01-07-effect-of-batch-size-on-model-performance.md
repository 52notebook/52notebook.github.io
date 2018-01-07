---
layout: post
title: MiniBatch size对模型性能的影响
description: "MiniBatch size既能提高数据并行化处理效率，又能影响模型性能，可见设置一个合适的batch size的重要性。"
modified: 2018-01-07T22:00:45-04:00
tags: [SGD, MiniBatch, 性能优化]
---


>论文 ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA

神经网络训练问题是**非凸优化**的重要内容，可用以下公式表示:

$$min_{x \in R^n}  f(x) := \frac{1}{M}\sum_{i=1}^Mf_i(x)$$
<!-- more -->

使用**SGD**或者其变异形式方法最下化损失函数。多次迭代最小化损失函数f:

$$x_{k+1} = x_k - \alpha_k(\frac{1}{|B_k|}\sum\nabla f_i(x_k))$$

$$B_k$$是batch sample，$$\alpha_k$$是第$$k$$步的步长。
这种优化方法称为带有噪音的**SGD**(通常，我们提到的**SGD**都是指**MiniBatch SGD**)。


这些优化方法具有以下性质:

- 收敛至**最小值处**(目标函数是凸函数时)，或者收敛到非凸函数的**驻点**
- 规避鞍点
- 对输入数据的增强
 
然**SGD**一个大的缺陷是**数据并行化限制**。为改善**SGD**并行化问题，一个可行的方案是增大**batch size**大小。

虽然增大**batch size**能提高数据并行化处理效率，但据试验观察训练得到的模型性能有小浮动地下降。这种模型泛化衰减又被称为**generalization gap**，最大可达5%。

究其原因，**generalization gap**是由于**large batch**方法使目标函数收敛在**sharp minimizers**处。

而**small batch size**之所以成功，在于**noisy gradient发挥了作用**

>it appears that noise in the gradient pushes the iterates out of the basin of attraction of sharp minimizers and encourages movement towards a flatter minimizer where noise will not cause exit from that basin. When the batch size is greater than the threshold mentioned above, the noise in the stochastic gradient is not sufficient to cause ejection from the initial basin leading to convergence to sharper a minimizer.

翻译总结是在**SGD**中，Noisy的存在使目标函数"推离"sharp minimizers，鼓励收敛在flat minimizers；但**batch size**大于某一阈值，这种作用变得不再有效。


**题外话**

梯度下降可看成是对损失函数的线性近似，沿着近似损失函数曲线向下移动。如果损失函数是非线性的，这种近方式似变得不再那么有效，因此往往需要减小**学习步伐$$\alpha$$**。

当**minibatch size**是$$M$$时，其需要$$O(M)$$大小的内存,但能降低的不确定信息量仅为$$O(\sqrt(M))$$。如果再继续增加**MiniBatch size**大小，则会出现**边际报酬递减**。

即便使用全部的训练集也未必能得到真实的梯度，因为真实梯度是关于所有可能的样本的梯度期望，输入数据与输出数据间的概率关系是未知的，不可能计算出真实梯度(再者**minibatch size**设置成全部数据集大小，很大的开销花在加载数据上，也并非能加速模型的训练)。


附:
$$r_{expect \space risk} = \int l(h(x;w),y) \space dP(x,y) \space = E[\space l(h(x;w),y) \space dP(x,y)]$$
$$P(x,y)$$是输入与输出数据间的真实概率分布关系。

然而，在实际应用中，目标函数简化成:

$$r_{empiriral \space risk} = \frac{1}{n}\sum_{i=1}^nl(h(x_i;w),y_i) $$
> 引用 http://www.deeplearningbook.org/contents/optimization.html(chapter 8)
>
> Optimization Methods for Large-Scale Machine Learning(p-14)
