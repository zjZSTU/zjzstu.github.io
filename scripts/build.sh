#!/bin/bash

set -eux

echo $PATH
export PATH=$PATH:$NODEJS_HOME

cd ./blogs/
rm node_modules/kramed/lib/rules/inline.js
cp inline.js node_modules/kramed/lib/rules/

npm run cg
hexo algolia