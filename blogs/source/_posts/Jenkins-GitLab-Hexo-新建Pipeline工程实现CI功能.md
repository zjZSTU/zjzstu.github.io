---
title: '[Jenkins][GitLab][Hexo]新建Pipeline工程实现CI功能'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: f80ec296
date: 2019-10-21 10:23:32
---

之前通过`Jenkins Freesstyle`工程实现了`Hexo`网站的`CI`部署，`Jenkins`还提供了`Pipeline`方式，能够更好的模块化构建过程

1. `Jenkins Pipeline`工程配置
2. `GitLab WebHook`配置
3. `Jenkinsfile`脚本编辑

## Jenkins Pipeline工程配置

新建工程`Hexo_Pipeline`，选择`Pipeline`类型

![](/imgs/jenkins-gitlab-hexo-pipeline/project-pipeline.png)

在配置页面，类别`Build Triggers`中选择构建`GitLab`

![](/imgs/jenkins-gitlab-hexo-pipeline/build-trigger.png)

在类别`Pipeline`中定义`Jenkinsfile`脚本来自于`git`工程，并输入`GitLab`项目地址

![](/imgs/jenkins-gitlab-hexo-pipeline/pipeline.png)

## GitLab WebHook配置

在`GitLab`项目中选择`Settings -> Integrations`，输入`WebHook URL`

## Jenkinsfile脚本编辑

在工程根目录新建文件`Jenkinsfile`

```
pipeline {
    agent any

    stages {
        stage('Install') {
            steps {
                echo 'Installing..'
                sh 'scripts/install.sh'
            }
        }
        stage('Build') {
            steps {
                echo 'Building..'
                sh 'scripts/build.sh'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
                sh 'scripts/deploy.sh'
            }
        }
    }
}
```

分`3`个阶段实现`CI`，脚本放置在`scripts`目录下

**注意：每个阶段的起始地址均是根目录**