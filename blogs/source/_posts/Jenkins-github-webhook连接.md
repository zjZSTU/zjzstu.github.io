---
title: '[Jenkins][github]webhook连接'
categories: 
- 工具
- 自动化
tags: jenkins
abbrlink: 341b6b1e
date: 2019-03-25 15:01:14
---

默认`Jenkins`已安装好`github`插件[Github plugin](http://wiki.jenkins-ci.org/display/JENKINS/Github+Plugin)

![](/imgs/Jenkins-github-webhook连接/github-plugin.png)

使用`WebHook`方式进行`github`的配置，过程如下：

1. 获取`Jenkins WebHook URL`
2. 配置`github`仓库`WebHook`
3. 新建`jenkins`工程并配置`github`仓库
4. 推送修改到`github`，触发`jenkins`工程

## 获取`Jenkins WebHook URL`

点击左侧菜单栏`->Manage Jenkins->Configure System`，在`GitHub`小节点击`Advanced`选项`->Override Hook URL`

![](/imgs/Jenkins-github-webhook连接/github-webhook.png)

## 配置`github`仓库`WebHook`

进入`github`仓库`Settings`页面，选择`Webhooks->Add webhook`，添加`URL`，`Content type`选择`application/json`格式

![](/imgs/Jenkins-github-webhook连接/add-webhook.png)

## 新建`jenkins`工程并配置

新建`Freestyle`工程`github_test`，进入配置页面

在`General`小节，添加`Github project`地址

![](/imgs/Jenkins-github-webhook连接/config-general.png)

在`Source Code Management`小节，选择`Git`并添加`Github project`地址（*相同就好了*）

![](/imgs/Jenkins-github-webhook连接/config-source-code-management.png)

在`Build Triggers`小节，选择`GitHub hook trigger for GITScm polling`选项

![](/imgs/Jenkins-github-webhook连接/config-build-triggers.png)

在`Build`小节，添加脚本如下

```
# 输出信息
echo "hello github"
# 当前路径
pwd
# 当前文件信息
ls -al
```

最后点击`Save`按钮

## 触发构建

`Jenkins`工程`github_test`的控制台输出如下

```
。。。
。。。
[github_test] $ /bin/sh -xe /tmp/jenkins5404470649999101259.sh
+ echo hello github
hello github
+ pwd
/home/ubuntu/.jenkins/workspace/github_test
+ ls -al
total 96
drwxrwxr-x  3 ubuntu ubuntu  4096 3月  25 14:56 .
drwxrwxr-x 15 ubuntu ubuntu  4096 3月  25 14:47 ..
-rw-rw-r--  1 ubuntu ubuntu     7 3月  25 14:56 coding.txt
drwxrwxr-x  8 ubuntu ubuntu  4096 3月  25 14:56 .git
-rw-rw-r--  1 ubuntu ubuntu    11 3月  25 14:56 github.txt
-rw-rw-r--  1 ubuntu ubuntu    14 3月  25 14:56 .gitignore
-rw-rw-r--  1 ubuntu ubuntu   492 3月  25 14:56 .gitmessage
-rw-rw-r--  1 ubuntu ubuntu    15 3月  25 14:56 hello.txt
-rw-rw-r--  1 ubuntu ubuntu     0 3月  25 14:56 hihihi.txt
-rw-rw-r--  1 ubuntu ubuntu    33 3月  25 14:56 hi.txt
-rw-rw-r--  1 ubuntu ubuntu 57146 3月  25 14:56 package-lock.json
-rw-rw-r--  1 ubuntu ubuntu   100 3月  25 14:56 README.md
Finished: SUCCESS
```
