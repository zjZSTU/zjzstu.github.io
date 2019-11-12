---
title: '[Jenkins][Nginx]反向代理'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: 7c823af7
date: 2019-10-27 15:22:21
---

当前`jenkins`通过`tomcat`进行托管，登录路径为

```
localhost:8080/jenkins/
```

下面通过`nginx`进行反向代理，简化登录路径

## nginx配置

修改`nginx`配置文件`/etc/nginx/conf.d/default.conf`，增加

```
...
...
    location /jenkins/ {
	    proxy_pass http://localhost:8080;
    }
...
...
```

刷新`nginx`

```
$ sudo nginx -t
$ sudo nginx -s reload
```

## 使用

配置完`nginx`后，就可以使用以下`URL`进行登录

```
http://localhost/jenkins/
```