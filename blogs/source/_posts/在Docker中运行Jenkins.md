---
title: 在Docker中运行Jenkins
categories:
  - - 工具
  - - 自动化
tags:
  - jenkins
  - docker
abbrlink: 202ee452
date: 2019-11-08 19:04:40
---

打算在远程服务器上运行`Jenkins`，忽然发现`git`没有安装，搞了半天没有成功（*各种依赖问题，条件限制不能重启机器*），所以尝试通过`Docker`运行`Jenkins`

## 完整命令

下面首先提供完整执行命令，再依次介绍其中`Jenkins`配置

```
$ docker run -d \
  --restart=always \
  -p 7070:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  --name jenkins \
  jenkins/jenkins
```

## 镜像

`Jenkins`提供了官方镜像 - [jenkins/jenkins](https://hub.docker.com/r/jenkins/jenkins/)

* 使用稳定版：`docker pull jenkins/jenkins:lts`
* 使用最新版：`docker pull jenkins/jenkins`

## 启动jenkins

`docker`命令如下：

```
$ docker run -p 7070:8080 -p 50000:50000 jenkins/jenkins
```

其可通过浏览器登录：`http://192.xx.xx.xx:7070/`

### 保存数据

`Jenkins`所有配置数据保存在路径`/var/jenkins_home`，可以使用一个卷保存配置数据，方便复用和移植操作

```
$ docker run -p 7070:8080 -p 50000:50000 -v jenkins_home:/var/jenkins_home jenkins/jenkins
```

`docker`会生成一个新卷`jenkins_home`，可通过`docker volume ls`查看

```
$ docker volume ls
DRIVER              VOLUME NAME
local               jenkins_home
```

### 后台运行

添加参数`-d`，设置容器在后台运行

```
$ docker run -d -p 7070:8080 -p 50000:50000 -v jenkins_home:/var/jenkins_home jenkins/jenkins
```

可通过命令`docker logs CONTAINER_ID`查询输出日志（比如初始密码）

```
$ docker logs b79
Running from: /usr/share/jenkins/jenkins.war
webroot: EnvVars.masterEnvVars.get("JENKINS_HOME")
...
...
```

## 设置执行器数量

`jenkins`默认允许`2`个执行器，可通过`groovy`脚本设置。新建脚本`executors.groovy`如下:

```
import jenkins.model.*
Jenkins.instance.setNumExecutors(5)
```

新建`Dockerfile`

```
FROM jenkins/jenkins:lts
COPY executors.groovy /usr/share/jenkins/ref/init.groovy.d/executors.groovy
```

重新编译生成新的`Jenkins`镜像

## 镜像升级

所有数据均保存在`/var/jenkins_home`，通过上节卷的方式保存数据，再次执行`docker pull jenkins/jenkins`即可升级到最新的`Jenkins`