#!/bin/bash

# 确保脚本在遇到任何错误时停止执行
set -e

# 执行第一个脚本
# echo "Running script1.sh"
# ./script/eval_iql_baseline.sh

# echo "Running script3.sh"
# ./script/eval_iql_no_matrix.sh

echo "Running script4.sh"
./script/eval_iql_no_explore.sh
  
echo "Running script5.sh"
./script/eval_mappo_baseline.sh

echo "Running script6.sh"
./script/eval_ippo_baseline.sh

# 执行第二个脚本
echo "Running script2.sh"
./script/eval_iql_apex.sh

# 执行第三个脚本
echo "Running script7.sh"
./script/eval_iql_CPG.sh

echo "All scripts executed successfully."
