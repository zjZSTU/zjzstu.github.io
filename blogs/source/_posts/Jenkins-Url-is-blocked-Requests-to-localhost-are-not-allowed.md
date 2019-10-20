---
title: '[Jenkins]Url is blocked: Requests to localhost are not allowed'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: 5d46d7f9
date: 2019-10-19 20:45:15
---

`GitLab`默认不允许本机`URL`进行`WebHook`连接

![](/imgs/jenkins-localhost/jenkins-localhost.png)

参考文章[Webhooks and insecure internal web services](https://docs.gitlab.com/ee/security/webhooks.html)，发现`GitLab`默认关闭了本机连接

参考文章[解决 Url is blocked: Requests to the local network are not allowed](https://www.cnblogs.com/zhongyuanzhao000/p/11379098.html)

1. 登录管理员账户
2. 进入`Configure GitLab -> Settings -> Network`，允许`local network`进行`web hooks and services`请求
3. 保存修改

![](/imgs/jenkins-localhost/outbound-request.png)