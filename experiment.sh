#!/bin/bash

# 创建实验日志目录
if [ ! -d "experiments" ]; then
  mkdir experiments
fi

# 第一组实验: 不同fusion方法
for f in concat weighted attention attention_alt; do
  echo "Running experiment with fusion=$f, ablation=none"

  # 修改配置文件
  sed -i "s/fusion: .*/fusion: \"$f\"/" config.yaml
  sed -i "s/ablation: .*/ablation: \"none\"/" config.yaml

  # 运行训练脚本并记录日志
  python3 train.py > experiments/fusion_${f}_ablation_none.log 2>&1
  python3 evaluate.py
done

# 第二组实验: 单模态实验
for a in text_only image_only; do
  echo "Running experiment with fusion=none, ablation=$a"

  # 修改配置文件
  sed -i "s/fusion: .*/fusion: \"none\"/" config.yaml
  sed -i "s/ablation: .*/ablation: \"$a\"/" config.yaml

  # 运行训练脚本并记录日志
  python train.py > experiments/fusion_none_ablation_${a}.log 2>&1
  python3 evaluate.py
done

echo "All experiments completed!"