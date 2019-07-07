---
title: '[Jenkins][ssh]coding连接'
categories: 
- 工具
- 自动化
tags: jenkins
abbrlink: 6185d82f
date: 2019-03-25 10:12:53
---

参考：[使用 Jenkins 构建 Coding 项目](https://open.coding.net/ci/jenkins/)

使用步骤如下：

1. 在`Jenkins`安装`coding`插件
2. 在`Jenkins`配置`Credentials`，设置`ssh`私钥
3. 新建工程，配置`coding`仓库地址以及`coding`触发器
4. 在`coding`仓库设置`webhook`
5. 推送修改到`coding`仓库，触发`jenkins`构建

## 安装`Coding Webhook Plugin`

默认没有安装`coding`插件，点击左侧菜单栏`->Manage Jenkins->Manage Plugins`

![](/imgs/Jenkins-ssh-coding连接/manage-plugins.png)

选择`Available`类别，在`Filter`框输入`coding`进行过滤，选中`Coding Webhook Plugin`后进行安装，重启

![](/imgs/Jenkins-ssh-coding连接/coding-plugin.png)

## 私钥设置

选择左侧菜单栏`->Credentials->System->Global credentials(unrestricted)`

![](/imgs/Jenkins-ssh-coding连接/global-credentials.png)

选择左侧菜单栏`->Add Credentials`

![](/imgs/Jenkins-ssh-coding连接/add-credentials.png)

`Kind`（类型）选择`SSH Username with privary key`，然后输入`Username`（自定义）和`privary key`（私钥）以及`Passphrase`（口令，如果有的话），点击`OK`按钮即可

![](/imgs/Jenkins-ssh-coding连接/create-ssh-credential.png)

## 工程配置

新建`Freestyle`工程`coding_test`，在`Source Code Management`(源码管理)部分配置`Git`仓库，同时添加之前设置的`credential`

![](/imgs/Jenkins-ssh-coding连接/source-code-management.png)

在`Build Triggers`(构建触发器)部分选择`Coding`构建（**在这里可以查询到`webhook url`**）

![](/imgs/Jenkins-ssh-coding连接/coding-trigger.png)

在`Build`(构建)部分添加脚本

```
# 输出信息
echo "hello coding"
# 当前路径
pwd
# 当前文件信息
ls -al
```

![](/imgs/Jenkins-ssh-coding连接/build-script.png)

最后点击`Save`按钮保存配置

## `webhook`设置

进入`Coding`仓库页面，选择设置`->WebHook`，点击新建`WebHook`按钮

![](/imgs/Jenkins-ssh-coding连接/coding-webhook.png)

添加`URL`，其他设置默认即可

## 触发构建

在本地下载`coding`仓库，修改后推送到`coding`仓库，`Jenkins`自动进行构建，控制台输出如下：

```
...
...
[coding_test] $ /bin/sh -xe /tmp/jenkins865426060185704403.sh
+ echo hello jenkins
hello jenkins
+ pwd
/home/ubuntu/.jenkins/workspace/coding_test
+ ls -al
total 96
drwxrwxr-x  3 ubuntu ubuntu  4096 3月  25 10:09 .
drwxrwxr-x 11 ubuntu ubuntu  4096 3月  25 09:40 ..
-rw-rw-r--  1 ubuntu ubuntu     7 3月  25 10:09 coding.txt
drwxrwxr-x  8 ubuntu ubuntu  4096 3月  25 10:09 .git
-rw-rw-r--  1 ubuntu ubuntu    11 3月  25 10:09 github.txt
-rw-rw-r--  1 ubuntu ubuntu    14 3月  25 10:09 .gitignore
-rw-rw-r--  1 ubuntu ubuntu   492 3月  25 10:09 .gitmessage
-rw-rw-r--  1 ubuntu ubuntu    15 3月  25 10:09 hello.txt
-rw-rw-r--  1 ubuntu ubuntu     0 3月  25 10:09 hihihi.txt
-rw-rw-r--  1 ubuntu ubuntu    33 3月  25 10:09 hi.txt
-rw-rw-r--  1 ubuntu ubuntu 57146 3月  25 10:09 package-lock.json
-rw-rw-r--  1 ubuntu ubuntu   100 3月  25 10:09 README.md
Finished: SUCCESS
```