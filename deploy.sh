#!/bin/bash

# abort on errors
set -e

git add .
git commit
git push

# build
yarn docs:build

# navigate into the build output directory
cd docs/.vuepress/dist

# if you are deploying to a custom domain
# echo 'www.example.com' > CNAME

git init
git add .
git commit -m 'New deployment'

git remote add origin git@github.com:ThiagoSoutoGit/Ubuntu.git

# if you are deploying to https://<USERNAME>.github.io
# git push -f git@github.com:<USERNAME>/<USERNAME>.github.io.git master

# if you are deploying to https://<USERNAME>.github.io/<REPO>
# git push -f origin master:gh-pages
git push -f origin master:gh-pages

cd -

yarn docs:devo
~