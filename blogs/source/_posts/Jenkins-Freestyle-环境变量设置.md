---
title: '[Jenkins][Freestyle]环境变量设置'
abbrlink: f2f14bee
date: 2019-10-20 19:18:46
categories:
  - - 工具
  - - 自动化
tags: jenkins
---

想要在`Freestyle`工程中设置加密的环境变量，使用插件`Environment Injector`完成

## 配置

首先下载/安装插件，进入`Manage Jenkins -> Manage Plugins -> Avaiable`，搜索插件`Environment Injector Plugin`并安装

![](/imgs/jenkins-env/env-injector-plugin.png)

## 使用

新建`Freestyle`工程，在`Build Environment`类别中

### 普通环境变量

选中`Inject environment variables to the build process`

![](/imgs/jenkins-env/inject-env.png)

在`Properties Content`输入键值对，就是构建过程可使用的环境变量

![](/imgs/jenkins-env/properties_content.png)

### 加密环境变量

选中`Inject passwords to the build as environment variables`

![](/imgs/jenkins-env/inject-passwd.png)

点击`Add`，输入`Name`和`Password`，就是构建过程中可使用的环境变量（已加密）

![](/imgs/jenkins-env/job-passwd.png)

## 测试

在脚本中打印环境变量

![](/imgs/jenkins-env/build-shell.png)

![](/imgs/jenkins-env/console-output.png)