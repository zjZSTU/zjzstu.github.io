---
title: Git工作流实践
categories:
  - - 版本管理
    - workflow
tags:
  - git
  - git-flow
  - gitlab-flow
  - github-flow
abbrlink: c7ee2f15
date: 2019-12-02 13:51:02
---

在好多个工程上都使用了`git`，随着时间的拉长会发现工程的提交历史和分支管理很混乱，所以希望能够有一套规范的`git`使用流程来更好的实现版本管理

参考[Git三大特色之WorkFlow(工作流)](https://blog.csdn.net/qq_32452623/article/details/78905181)，学习了目前最流行的三种`git`工作流

* [git flow](https://nvie.com/posts/a-successful-git-branching-model/)
* [github flow](http://scottchacon.com/2011/08/31/github-flow.html)
* [gitlab flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html)

## git flow

翻译地址：[[译]A successful Git branching model](https://www.zhujian.tech/posts/aae96086.html#more)

`git-flow`是最早发布的`git`工作流，其思路简洁清晰，通过`5`种分支类型即可管理整个工程

* `master`：主分支。保存后续生产环节的代码
* `develop`：开发分支。当实现功能足以反映下一版本的状态时，发布给`release`分支
* `feature`：特征分支。从`develop`分支`fork`，开发某个新特性，完成后合并回`develop`
* `release`：发布分支（或称为版本分支）。从`develop`分支`fork`，执行发布前的准备，包括小错误修复、版本元数据更新，最后合并到`master`和`develop`分支
* `hotfix`：热修复分支。从`master`分支`fork`，修复实时生产环节中出现的错误，完成后合并回`master`和`develop`分支

其中前两种分支属于主分支，长期存在于中间仓库中，后`3`种分支属于支持分支，完成所需要的目的后即可销毁

![](../imgs/git-workflow/git-model@2x.png)

## github flow

翻译地址：[[译]GitHub Flow](https://www.zhujian.tech/posts/a20843e9.html#more)

`github-flow`是`git-flow`的一个补充，提供了更加简单的工作流程。`github-flow`适用于持续部署的工程，直接将最新特征部署到`master`分支上，不再操作`develop`分支；同时通过`CI&CD`的使用，不再需要额外操作`release`分支和`hotfix`分支。`github`还结合了推送请求（`pull request`）功能，在合并`feature`分支之前通过`PR`请求其他成员对代码进行检查

* `master`分支中的任何东西都是可部署的
* 要开发新的东西，从`master`分支中创建一个描述性命名的分支(比如：`new-oauth2-scopes`)
* 在本地提交到该分支，并定期将您的工作推送到服务器上的同一个命名分支
* 当您需要反馈或帮助，或者您认为分支已经准备好合并时，可以提交一个推送请求（`PR`）
* 在其他人审阅并签署了该功能后，可以将其合并到`master`中
* 一旦它被合并并推送到`主服务器`，就可以并且应该立即部署

## gitlab flow

翻译地址：[[译]Introduction to GitLab Flow](https://www.zhujian.tech/posts/b35b83bc.html#more)

`gitlab-flow`出现的时间最晚，其内容更像是对前两者的补充，通过集成`git`工作流和问题跟踪系统，提供更加有序的关于环境、部署、发布和问题集成的管理

## 当前实践

当前操作工程大体分为两类：

1. 文档工程
2. 代码工程

对于文档工程而言，其主要内容即是文档内容以及文档工程框架，所以可以使用`github-flow`，直接将文档内容发布到`master`分支。而对于工程框架更新，可以新建`feature`分支，更新完成后再合并到`master`分支

而对于代码工程而言，其内容比较复杂，不仅包含功能实现的源代码，还有可能包含框架、文档等等，同时其使用环境比较不一致，所以根据具体环境进行操作