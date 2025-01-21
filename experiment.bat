@echo off
setlocal enabledelayedexpansion

:: 创建实验日志目录
if not exist "experiments" mkdir experiments

:: 第一组实验: 不同fusion方法
for %%f in (concat weighted attention attention_alt) do (
    echo Running experiment with fusion=%%f, ablation=none

    :: 修改配置文件
    powershell -Command "$content = Get-Content config.yaml; $content = $content -replace 'fusion: .*', 'fusion: \"%%f\"'; $content = $content -replace 'ablation: .*', 'ablation: \"none\"'; $content | Set-Content config.yaml"

    :: 运行训练脚本并记录日志
    python train.py > experiments/fusion_%%f_ablation_none.log 2>&1
    python evaluate.py
)

:: 第二组实验: 单模态实验
for %%a in (text_only image_only) do (
    echo Running experiment with fusion=none, ablation=%%a

    :: 修改配置文件
    powershell -Command "$content = Get-Content config.yaml; $content = $content -replace 'fusion: .*', 'fusion: \"none\"'; $content = $content -replace 'ablation: .*', 'ablation: \"%%a\"'; $content | Set-Content config.yaml"

    :: 运行训练脚本并记录日志
    python train.py > experiments/fusion_none_ablation_%%a.log 2>&1
    python evaluate.py
)

echo All experiments completed!