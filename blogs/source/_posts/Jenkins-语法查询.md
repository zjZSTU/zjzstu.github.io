---
title: '[Jenkins]语法查询'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: bf0708ea
date: 2020-03-10 14:57:58
---

`Jenkins`推荐使用`Pipeline`的方式进行工程构建，对于如何编写脚本，有`3`种方式可以参考

1. 在线文档
2. 流水线语法
3. 插件参考

下面的参考都是关于声明式流水线（`Declarative Pipeline`）的学习和使用

## 在线文档

最常使用的就是在线文档：[Pipeline](https://jenkins.io/doc/book/pipeline/)

## 流水线语法

这种方式是最近发现的，新建一个`Pipeline`工程，在配置页面的`流水线`类别中点击`流水线语法`

![](/imgs/jenkins-syntax/pipeline-syntax.png)

跳转到语法页面后，在右侧类别中选择`声明式指令生成器(Declarative Directive Generator)`，然后就可以在左侧根据需要选择具体的指令模板，比如需要配置`NodeJS`环境，之前已经在`全局工具配置`中设置了`NodeJS`节点`node`

1. 在Sample Directive中选择`tools: Tools`
2. 在NodeJS中选择`node`
3. 点击`Generate Declarative Directive`即可生成使用模板

```
tools {
  nodejs 'node'
}
```

![](/imgs/jenkins-syntax/declarative-directive-generator.png)

## 插件参考

在官网上查询指定插件，其描述页面会提供使用语法，以[NodeJS](https://plugins.jenkins.io/nodejs/)为例

## 小结

相比较而言，`3`种方式各有所长：

1. `在线文档`方式能够提供全面的语法参考
2. `流水线语法`方式能够进一步提供具体的操作模板
3. `插件参考`方式可以作为上面两种方式的补充