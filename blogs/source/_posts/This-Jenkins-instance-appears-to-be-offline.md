---
title: This Jenkins instance appears to be offline
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: 6af1c833
date: 2019-10-18 15:08:16
---

## 错误复现

重新安装`jenkins`，输入安装命令

```
$ java -jar jenkins.war --httpPort=8080 
```

在浏览器输入`localhost:8080`打开界面，输入密码后页面显示如下错误

```
This Jenkins instance appears to be offline
。。。
```

查看运行日志

```
2019-10-18 06:31:30.849+0000 [id=55]	INFO	hudson.util.Retrier#start: Calling the listener of the allowed exception 'connect timed out' at the attempt #1 to do the action check updates server
2019-10-18 06:31:30.851+0000 [id=55]	INFO	hudson.util.Retrier#start: Attempted the action check updates server for 1 time(s) with no success
2019-10-18 06:31:30.851+0000 [id=55]	SEVERE	hudson.PluginManager#doCheckUpdatesServer: Error checking update sites for 1 attempt(s). Last exception was: SocketTimeoutException: connect timed out
2019-10-18 06:31:30.853+0000 [id=55]	INFO	hudson.model.AsyncPeriodicWork$1#run: Finished Download metadata. 20,156 ms
2019-10-18 06:31:31.539+0000 [id=28]	WARNING	hudson.model.UpdateCenter#updateDefaultSite: Upgrading Jenkins. Failed to update the default Update Site 'default'. Plugin upgrades may fail.
...
...
```

问题在于`jenkins`无法正常升级

## 问题解决

参考：[Jenkins 2.89.3 “This Jenkins instance appears to be offline”](https://stackoverflow.com/questions/48726513/jenkins-2-89-3-this-jenkins-instance-appears-to-be-offline)

修改文件`~/.jenkins/hudson.model.UpdateCenter.xml`

```
...
<url>http://updates.jenkins.io/update-center.json</url>
...
```

将`https`修改为`http`。重新启动后恢复正常，进入插件下载页面

![](/imgs/jenkins-error/customize-jenkins.png)