#!/bin/bash

set -eux
cd ./blogs/

NEXT_DIR="themes/next"
CANVAS_DIR="${NEXT_DIR}/source/lib/canvas-nest"
ALGOLIA_DIR="${NEXT_DIR}/source/lib/algolia-instant-search"
FANCYBOX_DIR="${NEXT_DIR}/source/lib/fancybox"

NEXT_GIT="http://localhost:8800/zjzstu/hexo-theme-next.git"
CANVAS_GIT="http://localhost:8800/zjzstu/theme-next-canvas-nest.git"
ALGOLIA_GIT="http://localhost:8800/zjzstu/theme-next-algolia-instant-search.git"
FANCYBOX_GIT="http://localhost:8800/zjzstu/theme-next-fancybox3.git"

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

clone NEXT_DIR NEXT_GIT
clone CANVAS_DIR CANVAS_GIT
clone ALGOLIA_DIR ALGOLIA_GIT
clone FANCYBOX_DIR FANCYBOX_GIT

npm install