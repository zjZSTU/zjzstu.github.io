#!/bin/bash

set -eux

GIT_DIR="upload_git"
USER_NAME="zhujian"
USER_EMAIL="505169307@qq.com"

cd ./blogs/

if [ -e ${GIT_DIR} ]
then
    rm -rf ${GIT_DIR}
fi
mkdir ${GIT_DIR}
cd ${GIT_DIR}

git config --global user.name ${USER_NAME}
git config --global user.email ${USER_EMAIL}

git init
cp -r ../public/* ./
git add .
git commit -m "Update blogs"
git push --force git@148.70.133.9:/data/repositories/blogs.git master
git push --force git@github.com:zjZSTU/zjzstu.github.io.git master
git push --force git@git.dev.tencent.com:zjZSTU/zjZSTU.coding.me.git master