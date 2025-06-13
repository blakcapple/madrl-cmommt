#!/bin/bash

# 基础路径，这里需要根据你的具体情况进行修改
base_model_path="./load_model/iql_no_matrix/models"

# 循环从1到5，为seed变量赋值
for seed in {1..5}
do
   # 构建当前seed对应的模型文件名称
   model_file_name="mac_3_${seed}.pth"
   
   # 完整的模型文件路径
   model_file_path="${base_model_path}/${model_file_name}"

   echo "Copying model file $model_file_name to load_model directory"
   # 将指定的模型文件拷贝到load_model文件夹下面
   cp "$model_file_path" ./load_model/mac_3.pth

   echo "Running with seed $seed"
   # 运行python脚本，传入当前的seed值
   python start_eval.py --seed $seed --algo iql --learning_stage 3 --use_matrix
done

