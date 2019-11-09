
# zjzstu.github.io

[![Documentation Status](https://readthedocs.org/projects/zjzstugithubio/badge/?version=latest)](https://zjzstugithubio.readthedocs.io/zh_CN/latest/?badge=latest) [![Documentation Status](https://readthedocs.org/projects/hexo-guide/badge/?version=latest)](https://hexo-guide.readthedocs.io/zh_CN/latest/?badge=latest) [![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org) [![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

> 基于`Hexo`实现的个人博客网站

利用`Hexo`框架实现的个人博客网站 - https://www.zhujian.tech

网站制作包含以下服务：

* 网站框架：[Hexo](https://hexo.io)

* 主题：[NexT](https://github.com/zjZSTU/hexo-theme-next)

* 搜索：[Algolia](https://hexo-guide.readthedocs.io/zh_CN/latest/third-service/[Algolia]%E7%BD%91%E7%AB%99%E6%90%9C%E7%B4%A2.html)

* 评论系统：[Gitalk](https://hexo-guide.readthedocs.io/zh_CN/latest/third-service/[Gitalk]%E8%AF%84%E8%AE%BA%E7%B3%BB%E7%BB%9F.html)

* 访客和阅读次数统计：
    * [不蒜子](https://hexo-guide.readthedocs.io/zh_CN/latest/third-service/[%E4%B8%8D%E8%92%9C%E5%AD%90]%E6%96%87%E7%AB%A0%E9%98%85%E8%AF%BB%E6%AC%A1%E6%95%B0.html)
    * [百度统计](https://tongji.baidu.com/web/27249108/welcome/login)

* 持续集成：
    * [Travis CI](https://hexo-guide.readthedocs.io/zh_CN/latest/third-service/[Travis%20CI]%E6%8C%81%E7%BB%AD%E9%9B%86%E6%88%90.html)
    * [Jenkins](https://www.zhujian.tech/posts/446d640.html)

* 云服务：[腾讯云](https://cloud.tencent.com/?fromSource=gwzcw.1736893.1736893.1736893&gclid=Cj0KCQjwjYHpBRC4ARIsAI-3GkHCQESLZ49VY6v9zVtEgSVlnywvjdYO6VS7QN9Ia-vCQD1mQa0J8ywaAvdCEALw_wcB)

* 域名服务：[阿里云](https://wanwang.aliyun.com/domain/com/?spm=5176.10695662.1158081.1.59854234GbQWbo)

## 内容列表

- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

[从CSDN到Hexo](https://www.zhujian.tech/posts/359e7c3c.html)：

    从文档写作开始,经历了多个平台的实践
    最开始在CSDN上进行博客写作，到现在利用Hexo自建博客网站，中间还通过sphinx+github+readthedocs进行文档管理
    不同的写作平台和写作方式有长处也有短处，小结一下
    。。。

## 安装

博客文件编译需要预先安装以下工具：

```
$ npm install -g hexo-cli
$ sudo apt-get install git
```

文档文件编译需要预先安装以下工具：

```
$ pip install -U Sphinx
$ sudo apt-get install make
```

## 用法

### 博客编译

本地编译当前网站：

```
# 下载远程dev分支
$ mkdir zjzstu.github.io
$ cd zjzstu.github.io
$ git init
$ git remote add origin https://github.com/zjZSTU/zjzstu.github.io.git
$ git fetch origin dev
$ git checkout -b dev origin/dev

# 安装子模块
$ cd ./blogs/
$ git clone https://github.com/zjZSTU/hexo-theme-next.git themes/next
$ git clone https://github.com/zjZSTU/theme-next-canvas-nest themes/next/source/lib/canvas-nest
$ git clone https://github.com/zjZSTU/theme-next-algolia-instant-search themes/next/source/lib/algolia-instant-search
$ git clone https://github.com/zjZSTU/theme-next-fancybox3 themes/next/source/lib/fancybox

# 安装npm包
$ npm install

# 编译
$ npm run gg
```

完成后进入`blogs/public`目录，打开`index.html`文件

### 文档编译

浏览网站制作文档有两种方式

1. 在线浏览文档：[hexo指南](https://hexo-guide.readthedocs.io/zh_CN/latest/)
2. 本地生成文档，实现如下：

    ```
    $ cd zjzstu.github.io/docs
    $ make html
    ```
    编译完成后进入`docs/build/html`目录，打开`index.html`文件


## 主要维护人员

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjZSTU/zjzstu.github.com/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2019 zjZSTU
