# 如何使用 Real Instances 测试集

## 📖 概述

所有4个测试脚本现在都支持 `--real-instance-train` 参数，可以从真实实例的 train.csv 提取初始样本。

## 🚀 使用方法

### 基本语法

```bash
python <script_name>.py \
    --demand-file <path_to_test.csv> \
    --promised-lead-time 4 \
    --real-instance-train <path_to_train.csv>
```

### 示例 1: OR Baseline (or_csv_demo.py)

```bash
# 使用 real instance
python or_csv_demo.py \
    --demand-file real_instances_50_weeks/1047675/test.csv \
    --promised-lead-time 4 \
    --policy capped \
    --real-instance-train real_instances_50_weeks/1047675/train.csv

# 使用 synthetic instance (不提供 train.csv)
python or_csv_demo.py \
    --demand-file demand_case1_iid_normal.csv \
    --promised-lead-time 0 \
    --policy capped
```

### 示例 2: LLM Only (llm_csv_demo.py)

```bash
# 使用 real instance
python llm_csv_demo.py \
    --demand-file real_instances_50_weeks/168927/test.csv \
    --promised-lead-time 4 \
    --real-instance-train real_instances_50_weeks/168927/train.csv

# 使用 synthetic instance
python llm_csv_demo.py \
    --demand-file demand_case2_sudden_shift_cp15.csv \
    --promised-lead-time 2
```

### 示例 3: LLM to OR (llm_to_or_csv_demo.py)

```bash
# 使用 real instance
python llm_to_or_csv_demo.py \
    --demand-file real_instances_50_weeks/168989/test.csv \
    --promised-lead-time 4 \
    --policy capped \
    --real-instance-train real_instances_50_weeks/168989/train.csv

# 使用 synthetic instance
python llm_to_or_csv_demo.py \
    --demand-file demand_case3_increasing.csv \
    --promised-lead-time 0
```

### 示例 4: OR to LLM / Hybrid (or_to_llm_csv_demo.py)

```bash
# 使用 real instance
python or_to_llm_csv_demo.py \
    --demand-file real_instances_50_weeks/279137/test.csv \
    --promised-lead-time 4 \
    --policy capped \
    --real-instance-train real_instances_50_weeks/279137/train.csv

# 使用 synthetic instance
python or_to_llm_csv_demo.py \
    --demand-file demand_case4_normal_to_uniform_cp15.csv \
    --promised-lead-time 2
```

## 📊 批量测试所有 Real Instances

### Bash 版本

```bash
#!/bin/bash
# 批量测试所有 real instances

for instance in 1047675 168927 168989 172343 279137 521818 527757 827911 864511 938576
do
    echo "Testing instance: $instance"
    
    python or_csv_demo.py \
        --demand-file "real_instances_50_weeks/$instance/test.csv" \
        --promised-lead-time 4 \
        --policy capped \
        --real-instance-train "real_instances_50_weeks/$instance/train.csv" \
        > "output_or_${instance}.txt" 2>&1
    
    echo "Completed: $instance"
done
```

### PowerShell 版本

```powershell
# 批量测试所有 real instances

$instances = @("1047675", "168927", "168989", "172343", "279137", "521818", "527757", "827911", "864511", "938576")

foreach ($instance in $instances) {
    Write-Host "Testing instance: $instance" -ForegroundColor Cyan
    
    python or_csv_demo.py `
        --demand-file "real_instances_50_weeks/$instance/test.csv" `
        --promised-lead-time 4 `
        --policy capped `
        --real-instance-train "real_instances_50_weeks/$instance/train.csv" `
        > "output_or_${instance}.txt" 2>&1
    
    Write-Host "Completed: $instance" -ForegroundColor Green
}
```

## 🔍 参数说明

### `--real-instance-train` (新增)

- **类型**: 可选参数 (string)
- **默认值**: None
- **功能**: 
  - 当提供时: 从指定的 train.csv 读取 weeks 1-9 的需求数据作为初始样本
  - 未提供时: 使用默认的统一样本 `[108, 74, 119, 124, 51, 67, 103, 92, 100, 79]`
- **示例**: `--real-instance-train real_instances_50_weeks/1047675/train.csv`

### 其他参数保持不变

- `--demand-file`: 测试数据文件路径 (必需)
- `--promised-lead-time`: 承诺的交付周期 (默认: 0)
- `--policy`: OR 策略类型 (vanilla/capped, 默认: capped)
- `--human-feedback`: 启用人工反馈模式
- `--guidance-frequency`: 战略指导频率

## 📝 输出示例

使用 real instance 时，你会看到：

```
Using initial samples from real instance train.csv: real_instances_50_weeks/1047675/train.csv
  Samples (weeks 1-9): [56.0, 63.0, 106.0, 67.0, 74.0, 46.0, 58.0, 44.0, 63.0]
  Mean: 64.1, Count: 9
Promised lead time (used by OR algorithm): 4 days
```

使用 synthetic instance 时，你会看到：

```
Using default unified initial samples: [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
Promised lead time (used by OR algorithm): 0 days
```

## ⚠️ 注意事项

1. **Real instances 必须同时提供 test.csv 和 train.csv**
   - test.csv: 包含要测试的 50 周数据
   - train.csv: 包含初始化用的历史数据 (weeks 0-9)

2. **Synthetic instances 不需要 train.csv**
   - 使用默认的统一样本即可

3. **Real instances 参数固定**
   - lead_time: 4 天
   - profit: $2/单位
   - holding_cost: $1/单位/天

4. **文件路径**
   - 可以使用相对路径或绝对路径
   - Windows 用户注意路径分隔符

## 🎯 最佳实践

1. **对 real instances 始终使用 `--real-instance-train`**
   ```bash
   python or_csv_demo.py \
       --demand-file real_instances_50_weeks/1047675/test.csv \
       --real-instance-train real_instances_50_weeks/1047675/train.csv \
       --promised-lead-time 4
   ```

2. **对 synthetic instances 省略 `--real-instance-train`**
   ```bash
   python or_csv_demo.py \
       --demand-file demand_case1_iid_normal.csv \
       --promised-lead-time 0
   ```

3. **批量测试时重定向输出到文件**
   ```bash
   python or_csv_demo.py ... > output.txt 2>&1
   ```
