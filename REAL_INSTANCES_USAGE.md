# Real Instances Usage Guide

## 📊 新测试集结构说明

### 已完成的转换

新的测试集 `real_instances_50_weeks/` 包含10个真实场景实例，每个实例都已转换为与现有测试脚本兼容的格式。

### 转换后的文件结构

每个实例文件夹包含：
- **train.csv**: 训练数据（weeks 0-9），用于初始化 OR/LLM 算法
- **test.csv**: 测试数据（已转换为 weeks 1-50），用于实际测试

### test.csv 格式

转换后的 `test.csv` 文件包含以下列：

```csv
week,demand_{item_id},description_{item_id},lead_time_{item_id},profit_{item_id},holding_cost_{item_id},news
```

**示例** (168989 实例):
```csv
week,demand_168989,description_168989,lead_time_168989,profit_168989,holding_cost_168989,news
1,26.0,GROCERY I,4,2.0,1.0,42 weeks to Christmas
2,1.0,GROCERY I,4,2.0,1.0,41 weeks to Christmas
8,8.0,GROCERY I,4,2.0,1.0,"Holiday, National (35 weeks to Christmas)"
10,30.0,GROCERY I,4,2.0,1.0,"Additional, National; Event, National (33 weeks to Christmas)"
```

### 固定参数值

所有10个实例使用统一的参数：
- **lead_time**: 4 天
- **profit**: $2 / 单位
- **holding_cost**: $1 / 单位 / 天

### Item ID 映射

每个实例使用其文件夹名称作为 `item_id`：

| 文件夹 | Item ID | 描述 | 初始样本均值 |
|--------|---------|------|--------------|
| 1047675 | 1047675 | BEVERAGES | 64.1 |
| 168927 | 168927 | CLEANING | 74.2 |
| 168989 | 168989 | GROCERY I | 11.0 |
| 172343 | 172343 | GROCERY I | 24.9 |
| 279137 | 279137 | GROCERY I | 7.1 |
| 521818 | 521818 | GROCERY I | 18.1 |
| 527757 | 527757 | GROCERY I | 23.4 |
| 827911 | 827911 | GROCERY I | 15.0 |
| 864511 | 864511 | GROCERY I | 8.8 |
| 938576 | 938576 | GROCERY I | 13.2 |

## 🚀 如何使用

### 1. 使用现有测试脚本

所有4个测试脚本都可以直接使用新的测试集：

```bash
# LLM only
python llm_csv_demo.py --demand-file real_instances_50_weeks/1047675/test.csv --promised-lead-time 4

# OR only
python or_csv_demo.py --demand-file real_instances_50_weeks/168927/test.csv --promised-lead-time 4 --policy capped

# LLM to OR
python llm_to_or_csv_demo.py --demand-file real_instances_50_weeks/168989/test.csv --promised-lead-time 4

# OR to LLM (Hybrid)
python or_to_llm_csv_demo.py --demand-file real_instances_50_weeks/279137/test.csv --promised-lead-time 4
```

### 2. 初始样本数据

测试脚本需要修改以使用来自 `train.csv` 的真实初始样本，而不是统一的样本。

**当前代码** (在所有4个脚本中):
```python
# Generate initial demand samples for all items (unified across all products)
unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
```

**需要改为** (从 train.csv 读取):
```python
# Import at top of file
import pandas as pd
from pathlib import Path

# In main() function, after loading test.csv
# Extract initial samples from corresponding train.csv
test_path = Path(args.demand_file)
train_file = test_path.parent / "train.csv"

if train_file.exists():
    train_df = pd.read_csv(train_file)
    # Use weeks 1-9 from train.csv (exclude week 0 which is typically 0)
    train_samples = train_df[train_df['week_number'] >= 1]['demand'].tolist()
    initial_samples = {item_id: train_samples for item_id in csv_player.get_item_ids()}
    print(f"Using initial samples from train.csv: {train_samples}")
else:
    # Fallback to default samples if train.csv not found
    unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
    initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
    print(f"Train.csv not found, using default samples: {unified_samples}")
```

### 3. 所有10个实例的初始样本

如果需要硬编码所有初始样本：

```python
# Initial samples from train.csv (weeks 1-9) for all instances
REAL_INSTANCES_INITIAL_SAMPLES = {
    '1047675': [56.0, 63.0, 106.0, 67.0, 74.0, 46.0, 58.0, 44.0, 63.0],
    '168927': [106.0, 69.0, 110.0, 90.0, 70.0, 49.0, 53.0, 46.0, 75.0],
    '168989': [6.0, 41.0, 2.0, 6.0, 2.0, 3.0, 8.0, 24.0, 7.0],
    '172343': [39.0, 26.0, 30.0, 32.0, 33.0, 7.0, 24.0, 11.0, 22.0],
    '279137': [6.0, 7.0, 5.0, 11.0, 10.0, 5.0, 12.0, 3.0, 5.0],
    '521818': [34.0, 8.0, 7.0, 17.0, 23.0, 19.0, 14.0, 16.0, 25.0],
    '527757': [28.0, 16.0, 35.0, 32.0, 26.0, 14.0, 32.0, 11.0, 17.0],
    '827911': [11.0, 17.0, 15.0, 9.0, 11.0, 9.0, 22.0, 20.0, 21.0],
    '864511': [9.0, 3.0, 9.0, 10.0, 10.0, 6.0, 5.0, 11.0, 16.0],
    '938576': [18.0, 11.0, 23.0, 10.0, 7.0, 4.0, 10.0, 29.0, 7.0],
}

# Then in main():
item_id = csv_player.get_item_ids()[0]  # Get the item_id from CSV
if item_id in REAL_INSTANCES_INITIAL_SAMPLES:
    initial_samples = {item_id: REAL_INSTANCES_INITIAL_SAMPLES[item_id]}
else:
    # Fallback
    initial_samples = {item_id: [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]}
```

## 📝 需要修改的测试脚本

需要在以下4个脚本中更新初始样本读取逻辑：

1. ✅ `llm_csv_demo.py` (第509-510行)
2. ✅ `or_csv_demo.py` (第505-506行)
3. ✅ `llm_to_or_csv_demo.py` (类似位置)
4. ✅ `or_to_llm_csv_demo.py` (类似位置)

## 🔍 News 列说明

`news` 列合并了原始的 `holiday` 和 `weeks_to_christmas` 信息：

- **无假期**: `"42 weeks to Christmas"`, `"1 week to Christmas"`, `"Christmas week!"`
- **有假期**: `"Holiday, National (35 weeks to Christmas)"`, `"Additional, National; Event, National (33 weeks to Christmas)"`

这样 LLM 可以同时了解假期信息和距离圣诞节的时间。

## 🎯 测试建议

1. **先测试单个实例**: 选择一个实例（如 1047675）进行完整测试
2. **验证初始样本**: 确保从 train.csv 正确读取初始样本
3. **检查 lead_time**: 确认 promised_lead_time=4 正确传递
4. **批量测试**: 创建脚本循环测试所有10个实例
5. **对比结果**: 比较不同策略（LLM, OR, LLM-to-OR, OR-to-LLM）的表现

## 📊 转换脚本

- `transform_real_instances.py`: 转换 test.csv 格式
- `transform_train_to_initial_samples.py`: 提取初始样本数据

两个脚本都已成功运行并完成转换。
