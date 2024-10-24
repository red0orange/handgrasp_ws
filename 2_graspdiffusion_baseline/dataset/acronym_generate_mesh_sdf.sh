#!/bin/bash

# 遍历 0 到 8000，每次间隔 600
start=40
end=7900
interval=500

# 循环遍历区间
for ((i=$start; i<$end; i+=$interval)); do
    # 计算结束值
    next=$((i + interval))
    
    # 如果超过最大值，调整结束值为 end
    if [ $next -gt $end ]; then
        next=$end
    fi
    
    # 调用 Python 脚本，传递当前区间作为参数
    /home/red0orange/miniconda3/envs/3dapnet/bin/python /home/red0orange/Projects/handgrasp_ws/2_graspdiffusion_baseline/dataset/acronym_generate_mesh_sdf.py $i $next
done
