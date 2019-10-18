---
title: Jenkins 更换镜像源
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: 9ff7f63d
date: 2019-10-18 16:28:18
---

更新`Jenkins`国内镜像源，加速插件下载。参考[the status of Jenkins mirrors](http://mirrors.jenkins-ci.org/status.html)，目前国内有[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/jenkins/)

有两种配置方式

1. 文件配置
2. 页面配置

## 文件配置

修改文件`~/.jenkins/hudson.model.UpdateCenter.xml`

```
<?xml version='1.1' encoding='UTF-8'?>
<sites>
  <site>
    <id>default</id>
    <url>https://updates.jenkins.io/update-center.json</url>
  </site>
</sites>
```

添加国内地址

```
https://mirrors.tuna.tsinghua.edu.cn/jenkins/updates/update-center.json
```

重启应用即可

## 界面配置

参考[jenkins插件下载镜像加速](https://blog.csdn.net/you227/article/details/81076032)，进入`Manage Jenkins -> Manage Plugins -> Advanced`，修改`Update Site`