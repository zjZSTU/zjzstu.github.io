---
title: '[jenkinsci/blueocean][NodeJS]bash: node: command not found'
abbrlink: 5d3090aa
date: 2020-03-10 13:24:22
categories:
  - - 工具
  - - 自动化
  - [编程, 编程语言]
tags:
  - jenkins
  - nodeJS
---

在`Jenkins`官网教程[在Docker中下载并运行Jenkins](https://jenkins.io/zh/doc/book/installing/#%E5%9C%A8docker%E4%B8%AD%E4%B8%8B%E8%BD%BD%E5%B9%B6%E8%BF%90%E8%A1%8Cjenkins)中推荐使用`jenkinsci/blueocean`，运行后发现其对于`NodeJS`的支持并不完善

## NodeJS配置

由于之前已经使用镜像`jenkins/jenkins`，并保存在卷`jenkins_home`中，所以已经安装了插件`NodeJS`，具体操作参考[ [Jenkins]Pipeline工程配置NodeJS环境](https://www.zhujian.tech/posts/d521b4ea.html)

## env: ‘node’: No such file or directory

执行配置好的`Pipeline`工程，出现如下错误：

```
$ /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/npm install -g hexo gulp
env: ‘node’: No such file or directory
[Pipeline] envVarsForTool
$ /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/npm install -g hexo gulp
env: ‘node’: No such file or directory
```

## 问题解析

发现这个问题后，就手动登录了`jenkinsci/blueocean`镜像，在目录`/var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/`下找到了已下载好的`node/npm/npx`，执行命令时发现出错

```
bash-4.4# node
bash: node: command not found
```

即使配置环境变量`PATH`后仍旧会出错，在网上找到不少参考，发现这是个普遍的问题：

* [Jenkins - env: ‘node’: No such file or directory](https://stackoverflow.com/questions/51416409/jenkins-env-node-no-such-file-or-directory)
* [NPM unable to locate node binary Export](https://issues.jenkins-ci.org/browse/JENKINS-43593)

有人提议不使用`NodeJS`插件，而是通过手动安装`nodejs`的方式完成配置

```
apk add nodejs
```

不过尝试后发现也不靠谱，无法成功安装`NodeJS`，最后在网上发现一个讨论[Jenkins NodeJSPlugin node command not found](https://stackoverflow.com/questions/43307107/jenkins-nodejsplugin-node-command-not-found)，发现在`docker-apline`镜像中存在`node`依赖问题

```
bash-4.4# ldd /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node
	/lib64/ld-linux-x86-64.so.2 (0x7fbc31123000)
	libdl.so.2 => /lib64/ld-linux-x86-64.so.2 (0x7fbc31123000)
	libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x7fbc30fce000)
	libm.so.6 => /lib64/ld-linux-x86-64.so.2 (0x7fbc31123000)
	libgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x7fbc30fba000)
	libpthread.so.0 => /lib64/ld-linux-x86-64.so.2 (0x7fbc31123000)
	libc.so.6 => /lib64/ld-linux-x86-64.so.2 (0x7fbc31123000)
Error relocating /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node: gnu_get_libc_version: symbol not found
Error relocating /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node: __register_atfork: symbol not found
Error relocating /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node: secure_getenv: symbol not found
Error relocating /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node: setcontext: symbol not found
Error relocating /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node: makecontext: symbol not found
Error relocating /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node: backtrace: symbol not found
Error relocating /var/jenkins_home/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node: getcontext: symbol not found
```

## 解决方案

使用之前的镜像[jenkins/jenkins](https://hub.docker.com/r/jenkins/jenkins)

```
jenkins@5716bdc1c160:~/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin$ ldd node
	linux-vdso.so.1 (0x00007ffeb7dcb000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f00a6913000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f00a6591000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f00a628d000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f00a6076000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f00a5e59000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f00a5aba000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f00a6b17000)
jenkins@5716bdc1c160:~/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin$ ./node -v
v13.10.1
```