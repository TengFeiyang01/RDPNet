#!/bin/bash

# HPNet with RealMotion Integration 训练示例
# 结合 RealMotion 和 HPNet 的套娃训练

echo "开始 HPNet + RealMotion 套娃训练..."
echo "借鉴 RealMotion 的历史预测轨迹处理机制"

python train_realmotion.py \
    --root /path/to/your/argoverse/data \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --devices 1 \
    --max_epochs 64 \
    --historical_window 3 \
    --hidden_dim 128 \
    --num_historical_steps 20 \
    --num_future_steps 30 \
    --num_modes 6 \
    --num_attn_layers 3 \
    --num_heads 8 \
    --dropout 0.1 \
    --lr 3e-4 \
    --weight_decay 1e-4

echo "训练完成！"
echo "这个实现借鉴了 RealMotion 的："
echo "1. 历史预测轨迹坐标变换"
echo "2. memory_dict 缓存机制"
echo "3. 轨迹交互处理" 