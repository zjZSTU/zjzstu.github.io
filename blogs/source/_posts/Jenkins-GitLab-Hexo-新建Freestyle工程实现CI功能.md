---
title: '[Jenkins][GitLab][Hexo]新建Freestyle工程实现CI功能'
abbrlink: 446d640
date: 2019-10-20 19:49:12
categories: 
- [工具]
- [自动化]
tags: jenkins
---

之前通过`Travis CI`实现持续部署`Hexo`项目到腾讯云服务器，参考[[腾讯云][Travis CI]持续部署到云服务器](https://hexo-guide.readthedocs.io/zh_CN/latest/third-service/[腾讯云][Travis%20CI]持续部署到云服务器.html)

经过一段时间的使用，发现`Travis CI`传输文件到腾讯云服务器经常失败，所以打算在本地自建`Jenkins`，同时利用`GitLab`进行持续部署

实现步骤如下：

1. 关闭`Travis CI`触发器
2. 导入`Hexo`相关项目到`GitLab`
3. 新建`Jenkins Freestyle`工程

## 关闭Travis CI触发器

在`Travis CI`项目页面的`Settings`中关闭构建命令即可

![](/imgs/jenkins-gitlab-hexo-freestyle/travis-ci-settings.png)

## 导入Hexo相关工程到GitLab

本地`GitLab`的安装和配置参考[gitlab相关](https://zj-git-guide.readthedocs.io/zh_CN/latest/gitlab.html)

将`Hexo`相关工程导入到本地`GitLab`，有助于加速`Jenkins`构建

## 新建Jenkins Freestyle工程

`Jenkins`安装和配置参考[Jenkins使用指南](https://container-automation.readthedocs.io/zh_CN/latest/jenkins/index.html)

`Jenkins`中`GitLab`插件的配置以及`GitLab Webhook`的连接参考[[Jenkins][Gitlab]webhook连接](https://www.zhujian.tech/posts/6ff96ec3.html)

新建`Jenkins Freestyle`工程`hexo`，进入配置页面，构建脚本如下：

```
# 安装
cd ./blogs/
git clone http://localhost:8800/zjzstu/hexo-theme-next.git themes/next
git clone http://localhost:8800/zjzstu/theme-next-canvas-nest.git themes/next/source/lib/canvas-nest
git clone http://localhost:8800/zjzstu/theme-next-algolia-instant-search.git themes/next/source/lib/algolia-instant-search
git clone http://localhost:8800/zjzstu/theme-next-fancybox3.git themes/next/source/lib/fancybox
npm install
# 编译
rm node_modules/kramed/lib/rules/inline.js
cp inline.js node_modules/kramed/lib/rules/
npm run gs
hexo algolia
## 集成
mkdir upload_git
cd upload_git
git init
cp -r ../public/* ./
git add .
git commit -m "Update blogs"
git push --force git@148.70.xx.xx:/data/repositories/blogs.git master
git push --force git@github.com:zjZSTU/zjzstu.github.io.git master
git push --force git@git.dev.tencent.com:zjZSTU/zjZSTU.coding.me.git master
```

* 首先下载`GitLab`中的项目
* 然后编译生成`html`文件
* 最后上传到腾讯云服务器、`Gitlab和Coding`