---
title: '[Jenkins]Tomcat托管'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: bc77c204
date: 2019-10-22 11:34:20
---

参考：

[ubuntu16.04使用tomcat安装jenkins](http://www.mamicode.com/info-detail-2535358.html)

[[Ubuntu 16.02]Tomcat9安装](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tools/[Ubuntu%2016.02]Tomcat9%E5%AE%89%E8%A3%85.html)

通过`Tomcat`托管`Jenkins`，具体`Tomcat`操作参考

* [[Ubuntu 16.02]Tomcat9安装](https://zj-network-guide.readthedocs.io/zh_CN/latest/tomcat/[Ubuntu%2016.02]Tomcat9安装.html)
* [非root用户运行](https://zj-network-guide.readthedocs.io/zh_CN/latest/tomcat/[Ubuntu%2016.02]非root用户运行.html)

*当前`Tomcat`以普通用户`tomcat`身份运行*

## 实现

将`Jenkins.war`文件放置于`Tomcat webapps`目录下（*注意：设置`.war`文件的属主为`tomcat`*）

```
/opt/apache-tomcat-9.0.27/webapps
```

登录地址`localhost:8080/jenkins`，即可启动`Jenkins`

*`Tomcat`会在`webapps`目录下自动解压`Jenkins.war`，生成一个`jenkins`文件夹*

进入`Jenkins`页面后，修改`Manage Jenkins -> Configure System -> Jenkins Location`，修改`Jenkins URL`为相应的地址（*登录地址*），同时修改`GitLab`中`WebHook`地址

## Jenkins升级

下载新版本的`Jenkins.war`文件后，放置于`webapps`目录下，并删除`webapps/jenkins`文件夹，重新浏览器登录即可

## 修改主目录

如果`tomcat`以`root`用户运行，那么其相应的配置文件在`/root/.jenkins`目录下。修改`Jenkins`主目录在当前用户下 - `/home/zj/.jenkins`

### Tomcat配置

进入`apache tomcat`安装地址，新建`/bin/setenv.sh`，设置环境变量`JENKINS_HOME`

```
$ cat setenv.sh 
#!/bin/bash

export JENKINS_HOME=/home/zj/.jenkins
```

**注意`setenv.sh`的文件属性**

```
$ chown tomcat:tomcat setenv.sh
```

删除`Tomcat webapps`目录下的`jenkins`文件夹，重启`Tomcat`

### 查询

重新进行浏览器登录，在`Manage Jenkins -> Configure System`中查找`Home directory`

![](/imgs/jenkins-tomcat/jenkins-home-dir.png)

## 环境变量设置

由于`Tomcat`运行在其他普通用户下，所以还需要进一步将当前用户环境变量添加到`Jenkins`中，保证程序的执行（比如`node`）

参考：[jenkins执行脚本npm: command not found解决](https://blog.csdn.net/u011296165/article/details/96110294)

进入`Manage Jenkins -> Configure System`，在`Global properties`中选中`Environment variables`

![](/imgs/jenkins-tomcat/global-properties.png)