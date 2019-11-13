#!/bin/bash

set -eux

cd ./blogs/
rm node_modules/kramed/lib/rules/inline.js
cp inline.js node_modules/kramed/lib/rules/

npm run cg
hexo algolia