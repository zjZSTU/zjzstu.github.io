---
title: MkDocs vs Sphinx
categories:
  - - 随笔
  - - 工具
tags:
  - sphinx
  - mkdocs
abbrlink: 50a5fdf2
date: 2019-12-18 21:20:27
---

参考：[从CSDN到Hexo](https://zhujian.tech/posts/359e7c3c.html)

之前整理了一套文档生成、托管和发布流程，使用`Sphinx`完成工程文档的生成，使用`Github`完成文档的托管，使用`Readthedocs`完成文档的发布

在实践过程中发现整个流程都有或大或小的不足，尤其是`Sphinx`工具，最近学习了另外一个文档生成工具`MkDocs`，更加符合个人的需求

## Sphinx问题

`Sphinx`是一个文档生成工具，提供了方便快捷的文档生成操作，默认支持`reStructuredText`

虽然也能够使用`Markdown`，但是在实际操作过程中，发现若干问题：

1. 不能有效设置数学公式渲染
2. 不支持`Markdown`表格

## MkDocs解析

`MkDocs`同样是一个简单、易用的文档生成工具，其默认支持`Markdown`，能够使用表格语法，同时通过扩展能够解决数学公式渲染的问题

同时相比于`Sphinx`，其配置更加简洁易懂，降低使用门槛

## 小结

替换掉`Sphinx`，当前的文档操作流程是

>mkdocs文档制作，github远程托管，readthedocs在线发布 

实现教程：[MkDocs-Github-Readthedocs](https://zj-sphinx-github-readthedocs.readthedocs.io/en/latest/)