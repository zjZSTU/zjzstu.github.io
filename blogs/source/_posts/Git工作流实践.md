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

以上`3`种工作流各有所长，提供了不同工作环境下的分支策略和发布管理实践。结合自身实际需求，基于上述方法针对不同的项目进行调整

当前操作的工程大体分为三类：

1. 文档工程
2. 网站工程
3. 代码库工程

### 文档工程

文档工程指的是仓库存储的仅是文档以及文档生成框架，比如[zjZSTU/wall-guide](https://github.com/zjZSTU/wall-guide)和[zjZSTU/linux-guide](https://github.com/zjZSTU/linux-guide)，通常这些文档工程会结合`readthedocs`进行`CI&CD`的实现；有些文档工程会额外包含一些代码文件，比如[zjZSTU/Containerization-Automation](https://github.com/zjZSTU/Containerization-Automation)，里面包含了多个生成`docker`镜像的`dockerfiles`文件以及相关的配置文件

主要参考`GitHub Flow`，实现如下的分支策略：

1. 文档可直接发布到`master`分支
2. 要更新文档生成框架，需要创建特征分支，完成更新后再合并到`master`分支（**注意：合并前需要标记`master`分支，注明之前使用的版本**）
3. 每次添加新的代码功能，需要创建特征分支，完成更新后再合并到`master`分支
4. 如果需要修复已有的代码错误，直接在`master`分支上操作

### 网站工程

当前网站工程指的是基于`Hexo`的博客网站使用，博客网站主要包含`3`方面内容：

1. 网站框架
2. 主题
3. 文档

通常主题会在另一个仓库中管理，当前仓库中仅包含网站框架以及文档，同时还会包含`CI`配置文件

网上推荐的一种工作流比较简单：

1. 仅包含`master`和`dev`分支，其中`dev`分支存放工程源码，`master`分支存放编译后的`HTML`文件
2. 每次上传到`dev`分支后触发`CI`工具进行编译，并发送给`master`分支
3. `Web`服务器利用`master`分支进行网站发布

在实际操作中发现并不是每次上传到`dev`分支的修改都有必要通过`CI`工具进行编译，比如`README.md`；同时，在测试、更新`CI`工具和网站生成工具时，过多的调试不利于之后`log`的查询

参考`Git Flow`和`Github Flow`，实现如下的分支策略：

1. `master`分支存放编译好的网站文件
2. `dev`分支存放仓库源码
3. 文档文件直接上传到`dev`分支（*区分是否需要编译，通过`CI`文件进行判断*）
4. 要添加新的`CI`功能，需要创建特征分支，完成调试后再合并到`dev`分支
5. 要更新网站生成框架，需要创建特征分支，完成调试后再合并到`dev`分支（**注意：合并前需要标记`dev`分支，注明之前使用的版本**）
6. 修复`CI`配置文件或者网站配置文件错误，直接上传到`dev`分支

### 代码库工程

代码库工程是最常见的仓库类型，比如博客网站配套的主题仓库，[zjZSTU/PyNet](https://github.com/zjZSTU/PyNet/tree/dev)，[zjZSTU/GraphLib](https://github.com/zjZSTU/GraphLib/tree/dev)，里面不仅包含源代码，还有可能包含文档、图片、`CI`文件等等，对于个人操作而言，比较适用于`GitHub Flow`，通过快速的更新和迭代来完成开发