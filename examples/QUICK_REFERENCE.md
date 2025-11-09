# 快速参考：Real Instances vs Synthetic Instances

## 📌 命令对比

### Real Instances (新测试集)
```bash
python <script>.py \
    --demand-file real_instances_50_weeks/<instance>/test.csv \
    --promised-lead-time 4 \
    --real-instance-train real_instances_50_weeks/<instance>/train.csv
```

### Synthetic Instances (原测试集)
```bash
python <script>.py \
    --demand-file demand_case<X>_<name>.csv \
    --promised-lead-time <L>
```

## 📊 参数对比

| 参数 | Real Instances | Synthetic Instances |
|------|---------------|---------------------|
| `--demand-file` | `real_instances_50_weeks/*/test.csv` | `demand_case*.csv` |
| `--promised-lead-time` | `4` (固定) | 根据测试场景 (0, 2, 4等) |
| `--real-instance-train` | `real_instances_50_weeks/*/train.csv` (必需) | **不需要** (省略此参数) |
| `--policy` | `capped` (推荐) | `vanilla` 或 `capped` |

## 🎯 示例命令

### Real Instance 1047675 (BEVERAGES)
```bash
python or_csv_demo.py \
    --demand-file real_instances_50_weeks/1047675/test.csv \
    --promised-lead-time 4 \
    --policy capped \
    --real-instance-train real_instances_50_weeks/1047675/train.csv
```

### Synthetic Case 1 (IID Normal)
```bash
python or_csv_demo.py \
    --demand-file demand_case1_iid_normal.csv \
    --promised-lead-time 0 \
    --policy capped
```

## 📁 文件结构对比

### Real Instances
```
real_instances_50_weeks/
├── 1047675/
│   ├── train.csv  (weeks 0-9, 用于初始化)
│   └── test.csv   (weeks 1-50, 用于测试)
├── 168927/
│   ├── train.csv
│   └── test.csv
...
```

### Synthetic Instances
```
examples/
├── demand_case1_iid_normal.csv  (单文件，包含所有数据)
├── demand_case2_sudden_shift_cp15.csv
├── demand_case3_increasing.csv
...
```

## ⚡ 快速测试命令

### 测试所有4个策略 (Real Instance)
```bash
INSTANCE="1047675"
TEST="real_instances_50_weeks/$INSTANCE/test.csv"
TRAIN="real_instances_50_weeks/$INSTANCE/train.csv"

# 1. OR Baseline
python or_csv_demo.py --demand-file $TEST --promised-lead-time 4 --policy capped --real-instance-train $TRAIN

# 2. LLM Only
python llm_csv_demo.py --demand-file $TEST --promised-lead-time 4 --real-instance-train $TRAIN

# 3. LLM to OR
python llm_to_or_csv_demo.py --demand-file $TEST --promised-lead-time 4 --policy capped --real-instance-train $TRAIN

# 4. OR to LLM
python or_to_llm_csv_demo.py --demand-file $TEST --promised-lead-time 4 --policy capped --real-instance-train $TRAIN
```

## 💡 关键区别

| 特性 | Real Instances | Synthetic Instances |
|------|---------------|---------------------|
| 数据来源 | 真实零售数据 | 合成测试数据 |
| 初始样本 | 从 train.csv 动态读取 | 固定统一样本 |
| 测试周数 | 50 周 | 30-50 周 (不等) |
| News 信息 | 真实假期 + 距圣诞节周数 | 合成事件 |
| Lead Time | 固定为 4 | 可变 (0, 2, 4, 或动态变化) |
| Profit/Holding | 固定 (P=2, H=1) | 固定 (P=2, H=1) |

## 📝 初始样本来源

### Real Instances
```python
# 从 train.csv 读取 weeks 1-9
train_samples = train_df[train_df['week_number'] >= 1]['demand'].tolist()
# 例如: [56.0, 63.0, 106.0, 67.0, 74.0, 46.0, 58.0, 44.0, 63.0]
```

### Synthetic Instances
```python
# 固定的统一样本
unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
```
