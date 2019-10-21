#!/bin/bash

pwd
mkdir upload_git
cd upload_git
git init
cp -r ../public/* ./
git add .
git commit -m "Update blogs"
git push --force git@148.70.133.9:/data/repositories/blogs.git master
git push --force git@github.com:zjZSTU/zjzstu.github.io.git master
git push --force git@git.dev.tencent.com:zjZSTU/zjZSTU.coding.me.git master