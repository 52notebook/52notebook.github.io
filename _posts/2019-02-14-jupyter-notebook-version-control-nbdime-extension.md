---
layout: post
title: jupyter notebook版本控制扩展nbdime
description: "nbdime基于对notebook内容感知能力智能地合并冲突"
modified: 2018-02-14T17:00:45-04:00
tags: [版本控制, 插件, notebook]
---


在使用jupyter notebook开发过程中难免会遇到文件回退、历史文件对比等操作，如果手动对原始文件备份或者单纯靠记忆恢复历史版本总不容易想起细节问题，那么倘若Notebook可以像IDE那样使用版本控制就完美了。下面就介绍一种notebook 版本控制插件。
<!-- more -->

**nbdime** 基于对notebook内容感知差异和合并的能力智能地内容冲突。



- 差异对比展示
![nbdime](https://camo.githubusercontent.com/c8a53fe4eb0f5c8e7525d197e3870f5b3aaf7f64/68747470733a2f2f6e6264696d652e72656164746865646f63732e696f2f656e2f6c61746573742f5f696d616765732f6e62646966662d7765622e706e67)

- 安装&使用

```
# 安装nbdime
conda install -c conda-forge nbdime

# 安装jupyter extensions
jupyter serverextension enable --py nbdime  --user

jupyter nbextension install --py nbdime --user
jupyter nbextension enable --py nbdime --user
jupyter labextension install nbdime-jupyterlab
```

