---
title: '[Jenkins]Pipeline工程配置NodeJS环境'
categories:
  - - 工具
  - - 自动化
  - [编程, 编程语言]
tags:
  - jenkins
  - nodeJS
  - shell
abbrlink: d521b4ea
date: 2019-11-13 16:14:03
---

利用`Jenkins Pipeline`工程编译`NodeJS`项目，出现`npm not found`问题

参考[Jenkins Starting with Pipeline doing a Node.js test](https://medium.com/@gustavo.guss/jenkins-starting-with-pipeline-doing-a-node-js-test-72c6057b67d4)，配置`NodeJS`开发环境

## 插件

首先下载`NodeJS`插件，进入`Manage Jenkins -> Manage Plugins -> Available`，搜索`NodeJS`插件并安装

![](/imgs/jenkins-pipeline-nodejs/nodejs-plugin.png)

## 配置

插件安装完成后，进入`Manage Jenkins -> Global Tool Configuration`，会出现`NodeJS`的配置选项

![](/imgs/jenkins-pipeline-nodejs/nodejs-configure.png)

点击`NodeJS Insllation`，设置`Name`属性，选择要安装的`NodeJS`版本，以及待安装的全局软件，保存设置

![](/imgs/jenkins-pipeline-nodejs/nodejs-installation.png)

## Pipeline

新建Pipeline工程test，在配置时输入如下脚本

![](/imgs/jenkins-pipeline-nodejs/pipeline-script.png)

```
pipeline {
  agent any
  tools {nodejs "node"}
 
  stages {
    stage('Example') {
      steps {
        sh 'npm config ls'
        sh '''
            node -v
            npm -v
            gulp -v
            hexo -v
        '''
      }
    }
  }
}
```

*使用之前配置的`NodeJS`节点`node`*

执行结果如下：

```
Started by user zhujian
Running in Durability level: MAX_SURVIVABILITY
[Pipeline] Start of Pipeline
[Pipeline] node
Running on Jenkins in /home/zj/.jenkins/workspace/test
[Pipeline] {
[Pipeline] stage
[Pipeline] { (Declarative: Tool Install)
[Pipeline] tool
Unpacking https://nodejs.org/dist/v13.1.0/node-v13.1.0-linux-x64.tar.gz to /home/zj/.jenkins/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node on Jenkins
$ /home/zj/.jenkins/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/npm install -g hexo-cli gulp
npm WARN deprecated fsevents@1.2.9: One of your dependencies needs to upgrade to fsevents v2: 1) Proper nodejs v10+ support 2) No more fetching binaries from AWS, smaller package size
/home/zj/.jenkins/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/gulp -> /home/zj/.jenkins/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/lib/node_modules/gulp/bin/gulp.js
/home/zj/.jenkins/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/hexo -> /home/zj/.jenkins/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/lib/node_modules/hexo-cli/bin/hexo
npm WARN optional SKIPPING OPTIONAL DEPENDENCY: fsevents@2.1.2 (node_modules/hexo-cli/node_modules/fsevents):
npm WARN notsup SKIPPING OPTIONAL DEPENDENCY: Unsupported platform for fsevents@2.1.2: wanted {"os":"darwin","arch":"any"} (current: {"os":"linux","arch":"x64"})
npm WARN optional SKIPPING OPTIONAL DEPENDENCY: fsevents@1.2.9 (node_modules/gulp/node_modules/fsevents):
npm WARN notsup SKIPPING OPTIONAL DEPENDENCY: Unsupported platform for fsevents@1.2.9: wanted {"os":"darwin","arch":"any"} (current: {"os":"linux","arch":"x64"})

+ gulp@4.0.2
+ hexo-cli@3.1.0
added 382 packages from 508 contributors in 59.164s
[Pipeline] envVarsForTool
[Pipeline] }
[Pipeline] // stage
[Pipeline] withEnv
[Pipeline] {
[Pipeline] stage
[Pipeline] { (Example)
[Pipeline] tool
[Pipeline] envVarsForTool
[Pipeline] withEnv
[Pipeline] {
[Pipeline] sh
+ npm config ls
; cli configs
metrics-registry = "https://registry.npmjs.org/"
scope = ""
user-agent = "npm/6.12.1 node/v13.1.0 linux x64 ci/jenkins"

; node bin location = /home/zj/.jenkins/tools/jenkins.plugins.nodejs.tools.NodeJSInstallation/node/bin/node
; cwd = /home/zj/.jenkins/workspace/test
; HOME = /opt/tomcat
; "npm config ls -l" to show all defaults.

[Pipeline] sh
+ node -v
v13.1.0
+ npm -v
6.12.1
+ gulp -v
CLI version: 2.2.0
Local version: Unknown
+ hexo -v
hexo-cli: 3.1.0
os: Linux 4.15.0-64-generic linux x64
node: 13.1.0
v8: 7.8.279.17-node.19
uv: 1.33.1
zlib: 1.2.11
brotli: 1.0.7
ares: 1.15.0
modules: 79
nghttp2: 1.39.2
napi: 5
llhttp: 1.1.4
openssl: 1.1.1d
cldr: 35.1
icu: 64.2
tz: 2019a
unicode: 12.1
[Pipeline] }
[Pipeline] // withEnv
[Pipeline] }
[Pipeline] // stage
[Pipeline] }
[Pipeline] // withEnv
[Pipeline] }
[Pipeline] // node
[Pipeline] End of Pipeline
Finished: SUCCESS
```

在执行之前会先安装`NodeJS`，并安装预设置的应用