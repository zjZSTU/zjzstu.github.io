---
title: 如何编写好的README
categories: 
- [随笔]
- [版本管理]
tags: 
- readme
- git
abbrlink: 79f69ebe
date: 2019-09-10 19:06:02
---

在`github`上了上传了许多仓库，如何更好的管理、使用这些仓库，其中关键的一点在于`README`的编写。`README`的目的是向使用者展示仓库的使用方法、来历以及未来的进展。越来越重视写好一个`REAMDE`，*优秀的工程不一定有一个好的`README`，但是不好的`REAMDE`一定不是一个优秀的工程*

关于这个问题在网上也有许多讨论：[如何写好Github中的readme？](https://www.zhihu.com/question/29100816)。当前主要参考了一个关于如何编写标准`README`的`github`仓库：[RichardLitt/standard-readme](https://github.com/RichardLitt/standard-readme)

## 内容列表

下面首先讲解`standard-readme`提供了`README`文件编写规范，并结合网上讨论进行相应的调整，然后使用`README`生成器`yo`，最后编写一些适应不同场景的`README`模板

* [规范](#规范)
* [自定义](#自定义)
    * [中英文](#中英文)
    * [徽章](#徽章)
    * [版本更新日志](#版本更新日志)
    * [待办事项](#待办事项)
    * [参与贡献方式](#参与贡献方式)
    * [完整内容列表](#完整内容列表)
* [生成器](#生成器)
* [模板](#模板)

## 规范

`standard-readme`提供了一个编写规范，原文和翻译如下

* 原文：[Specification](https://github.com/RichardLitt/standard-readme/blob/master/spec.md)
* 翻译：[[译]规范](https://zj-git-guide.readthedocs.io/zh_CN/latest/readme/[译]规范.html)

里面提出了`README`的章节列表：

* 标题（`Title`，必须）
* 横幅（`Banner`，可选）
* 徽章（`Badges`，可选）
* 简短说明（`Short Description`，必须）
* 详细描述（`Long Description`，可选）
* 内容列表（`Table of Contents`，必须）
* 安全（`Security`，可选）
* 背景（`Background`，可选）
* 安装（`Install`，默认必须，对文档仓库而言可选）
* 用法（`Usage`，默认是必须的，对于文档仓库而言是可选的）
* 附加内容（`Extra Sections`，可选）
* 应用编程接口（`API`，可选）
* 主要维护人员（`Maintainers`，可选）
* 致谢（`Thanks`，可选）
* 参与贡献方式（`Contributing`，必须）
* 许可证（`License`，必须）

从上到下按需实现相应章节内容，其中横幅指的是仓库`logo`，内容列表指的是后续章节标题，而不是工程架构

### 横幅

网上有很多在线设计`logo`的网站，不过下载是要收费的

找了一个免费的`logo`设计网站：[logoly](https://www.logoly.pro/)

### 协议

参考[Adding a license to a repository](https://help.github.com/en/articles/adding-a-license-to-a-repository)

可以在线添加新文件，输入文件名为`LICENSE`或`LICENSE.md`，选择一个`license`模板，预览后提交即可

如果要更换协议，直接删除新建一个即可

## 自定义

`standard-readme`提供了相对完整的`README`章节架构，结合网上讨论和实际使用经验，再增加以下`4`个章节：

* 中英文
* 徽章
* 版本更新日志
* 待办事项

并更新章节参与贡献方式（`Contributing`）

### 中英文

* 状态：默认是必须的，对于文档仓库而言是可选的
* 必要条件：
    * `None`
* 建议：
    * 准备中英文两份`README`，相互之间可跳转
    * `README.md`为英文内容，`README.zh-CN.md`为中文内容
    * 放置在徽章之后，简短说明之前

### 徽章

* 状态：可选
* 必要条件：
    * 标题为徽章
* 建议：
    * 使用[http://shields.io](http://shields.io/)或类似服务创建和托管图像

规范中已经提到了一个徽章章节，里面添加的是仓库编写、部署过程中使用的工具、规范等相应的徽章，而本章节在于给出自己仓库的专属徽章

徽章生成参考[自定义徽章](https://zj-git-guide.readthedocs.io/zh_CN/latest/readme/自定义徽章.html)；章节内容参考[RichardLitt/standard-readme
](https://github.com/RichardLitt/standard-readme#badge)

### 版本更新日志

* 状态：可选
* 必要条件：
    * 使用`git`进行版本管理
* 建议：
    * 基于[Conventional提交规范](https://zj-git-guide.readthedocs.io/zh_CN/latest/message-guideline/Conventional提交规范.html)和[语义版本规范](https://zj-git-guide.readthedocs.io/zh_CN/latest/message-guideline/语义版本规范.html)进行版本提交
    * 使用[standard-version](https://zj-git-guide.readthedocs.io/zh_CN/latest/message-guideline/自动版本化和生成CHANGELOG工具standard-version.html)进行`CHANGELOG`生成
    * 链接到本地仓库的`CHANGELOG`文件

### 待办事项

* 状态：可选
* 必要条件：
    * `None`
* 建议：
    * 列出后续待完成的事项
    * 按实现顺序从上到下排列

### 参与贡献方式

在参与贡献方式章节中应该明确是否允许合并请求，并给出进行版本管理的相应规范，比如

* [git提交规范](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* [语义版本规范](https://zj-git-guide.readthedocs.io/zh_CN/latest/message-guideline/语义版本规范.html)
* [README编辑规范](https://github.com/RichardLitt/standard-readme)

### 完整内容列表

综上所述，完整的内容列表如下：

* 标题（`Title`，必须）
* 横幅（`Banner`，可选）
* 徽章（`Badges`，可选）
* 中英文(`Chinese and English`，默认是必须的，对于文档仓库而言是可选的)
* 简短说明（`Short Description`，必须）
* 详细描述（`Long Description`，可选）
* 内容列表（`Table of Contents`，必须）
* 安全（`Security`，可选）
* 背景（`Background`，可选）
* 徽章（`Badge`，可选）
* 安装（`Install`，默认必须，对文档仓库而言可选）
* 用法（`Usage`，默认是必须的，对于文档仓库而言是可选的）
* 附加内容（`Extra Sections`，可选）
* 应用编程接口（`API`，可选）
* 版本更新日志（`CHANGELOG`，可选）
* 待办事项（`TODO`，可选）
* 主要维护人员（`Maintainers`，可选）
* 致谢（`Thanks`，可选）
* 参与贡献方式（`Contributing`，必须）
* 许可证（`License`，必须）

## 生成器

参考：[README生成器](https://zj-git-guide.readthedocs.io/zh_CN/latest/readme/readme生成器.html)

## 模板

参考：

[README_TEMPLATES/](https://github.com/kylelobo/The-Documentation-Compendium/tree/master/en/README_TEMPLATES)

[example-readmes](https://github.com/RichardLitt/standard-readme/tree/master/example-readmes)

以仓库[zjZSTU/PyNet](https://github.com/zjZSTU/PyNet)为例，生成不同适用范围的自定义模板。`README`模板地址：[templates](https://github.com/zjZSTU/git-guide/tree/master/docs/source/readme/templates)

### 最小README

建立一个最简单基本的`README`，仅包含必须的章节内容，参考[MINIMAL_README.md](https://github.com/zjZSTU/git-guide/blob/master/docs/source/readme/templates/MINIMAL_README.md)

* 标题（`Title`，必须）
* 中英文(`Chinese and English`，默认是必须的，对于文档仓库而言是可选的)
* 简短说明（`Short Description`，必须）
* 内容列表（`Table of Contents`，必须）
* 安装（`Install`，默认必须，对文档仓库而言可选）
* 用法（`Usage`，默认是必须的，对于文档仓库而言是可选的）
* 参与贡献方式（`Contributing`，必须）
* 许可证（`License`，必须）

### 标准README

完整的实现所有章节内容，参考[STANDARD_README.md](https://github.com/zjZSTU/git-guide/blob/master/docs/source/readme/templates/STANDARD_README.md)