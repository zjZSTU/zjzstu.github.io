---
title: '[Jenkins]freestyle工程'
categories: 
- [工具]
- [自动化]
tags: jenkins
abbrlink: fddee3e1
date: 2019-03-23 10:14:45
---

`Jenkins`提供了多种模型来进行自动化操作，最基础的就是`freestyle`工程

操作步骤如下：

1. 在本地新建`git`仓库
2. 创建`Jenkins Freestyle`工程，绑定`git`仓库，执行构建脚本
3. `git`仓库添加文件
4. 手动触发`Jenkins`工程进行构建

## 新建`freestyle`工程

选择左侧菜单栏`->New Item`，输入项目名，选择工程类型为`Freestyle project`，点击`OK`按钮即完成新建

![](/imgs/Jenkins-freestyle工程/item-type-freestyle.png)

## 配置

完成新建项目后会跳往配置页面，也可在项目页面选择左侧菜单栏`->Configure`

配置页面有`6`个部分

1. `General`
2. `Source Code Management`
3. `Build Triggers`
4. `Build Environment`
5. `Build`
6. `Post-build Actions`

*配置完成点击`Save`或`Apply`按钮*

### 绑定`git`仓库

选择`Source Code Management->Git`，输入本地仓库地址

![](/imgs/Jenkins-freestyle工程/git-setting.png)

### 设置运行脚本

选择`Build->Add build step->Execute Shell`，使用`Linux`命令进行接下来的构建

![](/imgs/Jenkins-freestyle工程/build-script.png)

当前测试脚本如下

```
# 输出信息
echo "hello jenkins"
# 当前路径
pwd
# 当前文件信息
ls -al
```

## 构建

`git`仓库添加修改后，选择项目页面左侧菜单栏`->Build Now`，会在左侧下方`Build History`栏中显示新的构建，比如`#1`

点击本次构建，跳转到构建页面后选择左侧菜单栏`->Console Output`

```
。。。
。。。
[freestyle-test] $ /bin/sh -xe /tmp/jenkins191262132818038393.sh
+ echo hello jenkins
hello jenkins
+ pwd
/home/zj/.jenkins/workspace/freestyle-test
+ ls -al
总用量 16
drwxrwxr-x 3 zj zj 4096 3月  23 15:16 .
drwxrwxr-x 4 zj zj 4096 3月  23 15:16 ..
drwxrwxr-x 8 zj zj 4096 3月  23 15:22 .git
-rw-rw-r-- 1 zj zj   19 3月  23 15:16 hi.txt
Finished: SUCCESS
```