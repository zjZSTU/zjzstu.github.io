---
title: '[Jenkins][Gitlab]webhook连接'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: 6ff96ec3
date: 2019-10-19 19:47:06
---

完成`Jenkins+GitLab`的连接。步骤如下：

1. 申请`gitlab`私有`token`
2. 安装`jenkins for gitlab`插件
3. 在`jenkins`工程中配置`gitlab`
4. 在`gitlab`工程中配置`jenkins`

## 申请gitlab私有token

进入`Gitlab Settings -> Access Tokens`，输入`Name`，选择`api scopes`，生成私有访问`token`

## 安装jenkins for gitlab插件

进入`Manage Jenkins -> Manage Plugins -> Available`，选择`Gitlab`进行安装

![](/imgs/jenkins-gitlab/gitlab-plugin.png)

## 在jenkins工程中配置gitlab

首先进行全局配置，进入`Manage Jenkins -> Configure System`，输入`gitlab`主机`URL`和添加`GitLab`私有访问`token`

![](/imgs/jenkins-gitlab/jenkins-system-gitlab.png)

然后新建`Freestyle`工程，配置`gitlab`工程地址和触发器

![](/imgs/jenkins-gitlab/jenkins-scm.png)

![](/imgs/jenkins-gitlab/jenkins-trigger.png)

## 在gitlab工程中配置jenkins

在配置触发器时获取`WebHook URL`，在`gitlab`工程中进入`Settings -> Integrations`，输入`URL`进行配置

![](/imgs/jenkins-gitlab/gitlab-integrations.png)

完成配置后会在页面下方增加一个配置条目

![](/imgs/jenkins-gitlab/gitlab-webhook.png)

点击`Test -> Push events`，测试是否能够推送成功

## Hook executed successfully but returned HTTP 404

使用`localhost`进行登录，导致出现`404`错误，修改成局域网或者公网登录即可

## Hook executed successfully but returned HTTP 403

参考：[Hook executed successfully but returned HTTP 403](https://www.cnblogs.com/chenglc/p/11174530.html)

在Jenkins进入`Manage Jenkins -> Configure Global Security`

* 在`Access Control`类别下选中`Allow anonymous read access`
* 取消`CSRF Protection`类别下的`Prevent Cross Site Request Forgery exploits`

![](/imgs/jenkins-gitlab/access-control.png)

![](/imgs/jenkins-gitlab/csrf-protection.png)

进入`Manage Jenkins -> Gloabl System`，取消`Gitlab`类别下的`Enable authentication for '/project' end-point`

![](/imgs/jenkins-gitlab/gitlab-authentiation.png)
