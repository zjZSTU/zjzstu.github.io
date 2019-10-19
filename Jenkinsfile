#echo -e "Host 148.70.133.9\n\tStrictHostKeyChecking no\n" >> ~/.ssh/config
#npm install -g hexo-cli
# 安装
git submodule foreach git pull origin master
cd ./blogs/
git clone https://github.com/theme-next/theme-next-canvas-nest themes/next/source/lib/canvas-nest
git clone https://github.com/theme-next/theme-next-algolia-instant-search themes/next/source/lib/algolia-instant-search
git clone https://github.com/theme-next/theme-next-fancybox3 themes/next/source/lib/fancybox
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
git push --force git@148.70.133.9:/data/repositories/blogs.git master
cd ..
sed -i'' "s~https://github.com/zjZSTU/zjzstu.github.com.git~https://${GITHUB_REPO_TOKEN}@github.com/zjZSTU/zjzstu.github.com.git~"
  _config.yml
sed -i'' "s~git@git.dev.tencent.com:zjZSTU/zjZSTU.coding.me.git~https://zjZSTU:${CODING_REPO_TOKEN}@git.coding.net/zjZSTU/zjZSTU.coding.me.git~"
  _config.yml
hexo deploy