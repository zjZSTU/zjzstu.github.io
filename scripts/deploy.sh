#!/bin/bash

set -eux

# GIT环境

USER_NAME=`git config --global user.name`
USER_EMAIL=`git config --global user.email`

if [[ -z ${GIT_USER_NAME} ]]
then
	git config --global user.name ${GIT_USER_NAME}
fi

if [[ -z ${USER_EMAIL} ]]
then
    git config --global user.email ${GIT_USER_EMAIL}
fi

## SSH环境

if [[ -s ~/.ssh/config ]]
then
    touch ~/.ssh/config
fi

SSH_CONFIG=`cat ~/.ssh/config | grep ${HEXO_SERVER_ADDRESS}`

if [[ -z ${SSH_CONFIG} ]]
then
    echo -e "Host ${HEXO_SERVER_ADDRESS}\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
fi

## GIT操作

GIT_DIR="upload_git"

cd ./blogs/

if [ -e ${GIT_DIR} ]
then
    rm -rf ${GIT_DIR}
fi
mkdir ${GIT_DIR}
cd ${GIT_DIR}

git init
cp -r ../public/* ./
git add .
git commit -m "Update blogs"
git push --force git@${HEXO_SERVER_ADDRESS}:/data/repositories/blogs.git master
git push --force git@github.com:zjZSTU/zjzstu.github.io.git master
git push --force git@git.dev.tencent.com:zjZSTU/zjZSTU.coding.me.git master