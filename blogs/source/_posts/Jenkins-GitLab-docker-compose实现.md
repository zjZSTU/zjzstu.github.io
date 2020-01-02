---
title: '[Jenkins][GitLab]docker-compose实现'
abbrlink: 1431c640
date: 2020-01-01 23:14:24
categories: 
- [工具]
- [自动化]
- [版本控制, 托管平台]
tags: 
- jenkins
- gitlab
- docker
---

之前实现了[在Docker中运行Jenkins](https://zhujian.tech/posts/202ee452.html)以及[[Docker]GitLab使用](https://zj-git-guide.readthedocs.io/zh_CN/latest/platform/[Docker]GitLab%E4%BD%BF%E7%94%A8/)，参考[Docker Compose](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/compose/[%E8%AF%91]Docker%20Compose%E6%A6%82%E8%BF%B0/)，通过`docker-compose`方式同时启动两个容器

## docker-compose.yml

```
version: "3.7"
services: 
    jenkins:
        labels:
            AUTHOR: "zhujian <zjzstu@github.com>"
        container_name: jenkins
        image: jenkins/jenkins
        volumes: 
            - "jenkins_home:/var/jenkins_home"
        ports: 
            - "7070:8080"
            - "50000:50000"
        restart: always
        tty: true
        stdin_open: true
    gitlab:
        labels:
            AUTHOR: "zhujian <zjzstu@github.com>"
        container_name: gitlab
        image: gitlab/gitlab-ce:latest
        volumes: 
            - "/srv/gitlab/config:/etc/gitlab"
            - "/srv/gitlab/logs:/var/log/gitlab"
            - "/srv/gitlab/data:/var/opt/gitlab"
        ports: 
            - "7010:7010"
            - "7020:22"
        restart: always
        tty: true
        stdin_open: true
volumes: 
    jenkins_home:
        external: true
```

## 启动/停止

```
# 启动，后台运行
$ docker-compose up -d
# 停止并移除容器
$ docker-compose down
```

**注意：上述命令需要在`docker-compose.yml`路径下执行**