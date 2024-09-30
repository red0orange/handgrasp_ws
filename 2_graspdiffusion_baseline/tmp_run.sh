#!/bin/bash

# # Python脚本的路径
# PYTHON_SCRIPT="obtain_isaacgym_eval_cong_dataset.py"

# # 随机数的范围
# MIN=3062
# MAX=7720

# for (( i=MIN; i<=MAX; i++ ))
# do
#     echo "Trying to execute with random number: $i"
    
#     # 使用随机数作为参数执行Python脚本
#     /home/red0orange/miniconda3/envs/3dapnet/bin/python "$PYTHON_SCRIPT" ./data/grasp_CONG_graspldm/cong_isaacgym_eval_data.npy "$i"
# done

# # 生成随机数的函数
# generate_random_number() {
#     echo $(($RANDOM % $MAX + $MIN))
# }

# # 主执行函数
# execute_script() {
#     local random_number=$(generate_random_number)
#     echo "Trying to execute with random number: $random_number"
    
#     # 使用随机数作为参数执行Python脚本
#     /home/red0orange/miniconda3/envs/3dapnet/bin/python "$PYTHON_SCRIPT" ./data/grasp_CONG_graspldm/cong_isaacgym_eval_data.npy "$random_number"
    
#     # 检查Python脚本的退出状态
#     # if [ $? -ne 0 ]; then
#     echo "Execution failed with exit code $?"
#     execute_script # 如果失败，递归调用自身
#     # else
#     #     echo "Execution succeeded."
#     # fi
# }

# # 调用主执行函数
# execute_script

# 主执行函数
execute_script() {
    echo "Trying to execute with random number"
    
    # 使用随机数作为参数执行Python脚本
    /home/red0orange/miniconda3/envs/3dapnet/bin/python "obtain_isaacgym_eval_cong_dataset.py"
    
    # 检查Python脚本的退出状态
    # if [ $? -ne 0 ]; then
    echo "Execution failed with exit code $?"
    sleep 360
    execute_script # 如果失败，递归调用自身
    # else
    #     echo "Execution succeeded."
    # fi
}

# 调用主执行函数
execute_script