#!/bin/bash

set -eux

# algolia环境
export HEXO_ALGOLIA_INDEXING_KEY=${HEXO_ALGOLIA_KEY}

cd ./blogs/
rm node_modules/kramed/lib/rules/inline.js
cp inline.js node_modules/kramed/lib/rules/

npm run cg
hexo algolia