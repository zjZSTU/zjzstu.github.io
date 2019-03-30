---
title: jupyter-notebook远程登录
categories: server
tags: jupyter-notebook
abbrlink: 5e96fc4f
date: 2019-03-30 10:48:06
---

参考：

[Running a notebook server](https://jupyter-notebook.readthedocs.io/en/latest/public_server.html#notebook-server-security)

[设置 jupyter notebook 可远程访问](https://blog.csdn.net/simple_the_best/article/details/77005400)

`jupyter notebook`是一个基于客户端-服务器架构的`web`应用，但是默认仅能运行在本地，可以通过配置开放远程服务器端口

本文实现单用户远程访问功能，如果要实现多用户访问，参考[JupyterHub](https://jupyterhub.readthedocs.io/en/latest/)

## 生成配置文件

使用参数`--generate-config`生成配置文件`jupyter_notebook_config.py`，存储在`~/.jupyter`文件夹

```
$ jupyter notebook --generate-config
Writing default config to: /home/zj/.jupyter/jupyter_notebook_config.py
```

## 修改监听`ip`地址

修改服务器监听`ip`地址

```
## The IP address the notebook server will listen on.
#c.NotebookApp.ip = 'localhost'
```

默认为`localhost`（就是本地），允许所有`ip`地址访问

```
c.NotebookApp.ip = '*'
```

## 远程登录

在远程服务器修改完配置文件后，启动`jupyter`

```
$ jupyter notebook
[W 11:13:25.418 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[I 11:13:25.423 NotebookApp] Serving notebooks from local directory: /home/zj/software/tutorial
[I 11:13:25.423 NotebookApp] 0 active kernels 
[I 11:13:25.424 NotebookApp] The Jupyter Notebook is running at: http://[all ip addresses on your system]:8888/?token=e78e227ffb28ede99b5b7d576459335fd265e886d834ec32
[I 11:13:25.424 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 11:13:25.424 NotebookApp] No web browser found: could not locate runnable browser.
[C 11:13:25.424 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=e78e227ffb28ede99b5b7d576459335fd265e886d834ec32
[I 11:13:41.883 NotebookApp] 302 GET / (192.168.0.140) 0.94ms
[I 11:13:41.915 NotebookApp] 302 GET /tree? (192.168.0.140) 1.24ms
```

在本地浏览器输入`远程IP:8888`，就能进入页面，还需要输入`token`（*在输入日志中*）

![](/imgs/jupyter-notebook远程登录/jupyter-token-login.png)

## 密码设置

可以进行密码设置以加强安全性
