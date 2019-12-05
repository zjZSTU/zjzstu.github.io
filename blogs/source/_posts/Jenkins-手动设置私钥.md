---
title: '[Jenkins]手动设置私钥'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: c343c930
date: 2019-12-05 13:49:58
---

需要在`Jenkins`操作完成后上传代码到另一个网站的仓库，所以需要手动设置`credential`

## 自动验证

在`Jenkins`中添加`SSH Username with private key`类型的凭据后，就可以在配置git仓库的时候设置

![](/imgs/jenkins-credentials/credentials.png)

之后运行过程中`Jenkins`会自动通过该凭据进行`ssh`验证，下载`git`代码

## 手动验证

在`Freestyle`工程和`Pipeline`工程中进行配置如下

### Freestyle

新建`Freestyle`工程，在`配置 -> 构建环境`类别中选择`Use secret text(s) or files(s)`，新增一个`SSH User Private Key`

![](/imgs/jenkins-credentials/freestyle-env.png)

在`Key文件变量`中设置一个变量名，在`凭据`中选定之前设置的私钥

在构建环节，选择脚本执行，在操作时将私钥写入`.ssh`文件夹并设置文件权限

```
rm ~/.ssh/id_rsa
cat $TTE > ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa

git clone git@148.xx.xxx.x:/data/repositories/blogs.git

rm ~/.ssh/id_rsa
```

### Pipeline

参考：[Secret 文本，带密码的用户名，Secret 文件](https://jenkins.io/zh/doc/book/pipeline/jenkinsfile/#for-secret-text-usernames-and-passwords-and-secret-files)

新建`Pipeline`工程，在`配置 -> 流水线`中选择脚本操作，实现如下：

```
pipeline {
   agent any

  environment {
    id_rsa=credentials('fa1b8cd3-cxxx-4xxx-axx-2xxxa8dc5b4a')
  }

   stages {
      stage('Hello') {
         steps {
            echo 'Hello World'
            sh '''
            git --version
            cat ${id_rsa} > ~/.ssh/zj_id_rsa
            chmod 600 ~/.ssh/zj_id_rsa

            git clone ...        
            '''
         }
      }
   }
}
```

设置环境变量`id_rsa`，调用函数`credentials`提取已定义的私钥（标识号可在凭证中查询）

![](/imgs/jenkins-credentials/unique-label.png)
