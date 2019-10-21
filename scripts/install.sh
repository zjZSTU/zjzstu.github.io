#!/bin/bash

cd ./blogs/
pwd
git clone http://localhost:8800/zjzstu/hexo-theme-next.git themes/next
git clone http://localhost:8800/zjzstu/theme-next-canvas-nest.git themes/next/source/lib/canvas-nest
git clone http://localhost:8800/zjzstu/theme-next-algolia-instant-search.git themes/next/source/lib/algolia-instant-search
git clone http://localhost:8800/zjzstu/theme-next-fancybox3.git themes/next/source/lib/fancybox
npm install