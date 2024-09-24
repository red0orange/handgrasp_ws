#!/bin/bash

# 检查输入参数数量
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <commit_info_file> <uncommitted_changes_file>"
  exit 1
fi

# 获取输入参数
commit_info_file=$1
uncommitted_changes_file=$2

# 获取最新的 3 个 commit 的 id 和注释并保存到指定文件
git log -3 --format="%H%n%s%n%b%n" > "$commit_info_file"

# 获取未 commit 的修改内容（包括新增文件）并保存到指定文件
git diff > "$uncommitted_changes_file"
git diff --cached >> "$uncommitted_changes_file"
