#!/bin/bash

set -eux

cd ./blogs/

NEXT_DIR="themes/next"
CANVAS_DIR="${NEXT_DIR}/source/lib/canvas-nest"
ALGOLIA_DIR="${NEXT_DIR}/source/lib/algolia-instant-search"
FANCYBOX_DIR="${NEXT_DIR}/source/lib/fancybox"
PACE_DIR="${NEXT_DIR}/source/lib/pace"

NEXT_GIT="http://192.168.0.184:7010/zjZSTU/hexo-theme-next.git"
CANVAS_GIT="http://192.168.0.184:7010/zjZSTU/theme-next-canvas-nest.git"
ALGOLIA_GIT="http://192.168.0.184:7010/zjZSTU/theme-next-algolia-instant-search.git"
FANCYBOX_GIT="http://192.168.0.184:7010/zjZSTU/theme-next-fancybox3.git"
PACE_GIT="http://192.168.0.184:7010/zjZSTU/theme-next-pace.git"

function clone()
{
    DIR=$1
    GIT=$2
    if [ -e ${DIR} ]
    then
        rm -rf ${DIR}
    fi
    git clone ${GIT} ${DIR}
}

function next() 
{
    DIR=$1
    GIT=$2

    if [ -e ${DIR} ]
    then
        rm -rf ${DIR}
    fi
    mkdir ${DIR}
    cd ${DIR}
    git init
    git remote add origin ${GIT}
    git fetch origin dev
    git checkout -b dev origin/dev

    cd ../..
}

next ${NEXT_DIR} ${NEXT_GIT}
clone ${CANVAS_DIR} ${CANVAS_GIT}
clone ${ALGOLIA_DIR} ${ALGOLIA_GIT}
clone ${FANCYBOX_DIR} ${FANCYBOX_GIT}
clone ${PACE_DIR} ${PACE_GIT}

# 调用缓存包

Node_DIR="node_modules"
Node_GIT="http://192.168.0.184:7010/zjZSTU/node_modules.git"

if [ -e ${Node_DIR} ]
then
    rm -rf ${Node_DIR}
fi

git clone ${Node_GIT}
npm install