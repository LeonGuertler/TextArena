## ✅ 已完成的修改

### 1. test.csv 文件转换 (全部10个实例)

**转换内容**:
- ✅ Week number: 10-59 → 1-50
- ✅ 添加列: `demand_{item_id}`, `description_{item_id}` (item_id = 文件夹名)
- ✅ 添加列: `lead_time_{item_id}` = 4 (全部实例)
- ✅ 添加列: `profit_{item_id}` = 2.0 (全部实例)
- ✅ 添加列: `holding_cost_{item_id}` = 1.0 (全部实例)
- ✅ 合并列: `holiday` + `weeks_to_christmas` → `news` (带文字说明)
- ✅ 删除列: 原始的 `week_number`, `demand`, `description`, `holiday`, `weeks_to_christmas`

**转换结果**:
```
实例          Item ID    描述         初始样本数 (train.csv)
1047675      1047675    BEVERAGES    9个样本 (均值: 64.1)
168927       168927     CLEANING     9个样本 (均值: 74.2)
168989       168989     GROCERY I    9个样本 (均值: 11.0)
172343       172343     GROCERY I    9个样本 (均值: 24.9)
279137       279137     GROCERY I    9个样本 (均值: 7.1)
521818       521818     GROCERY I    9个样本 (均值: 18.1)
527757       527757     GROCERY I    9个样本 (均值: 23.4)
827911       827911     GROCERY I    9个样本 (均值: 15.0)
864511       864511     GROCERY I    9个样本 (均值: 8.8)
938576       938576     GROCERY I    9个样本 (均值: 13.2)
```

### 2. train.csv 分析完成

已提取所有10个实例的初始样本数据 (weeks 1-9)，可用于初始化 OR/LLM 算法。

---

## 🔧 还需要修改的内容

### 需要修改的4个测试脚本

所有测试脚本目前使用硬编码的统一样本，需要改为从 `train.csv` 读取真实初始样本：

#### 1. `llm_csv_demo.py`
#### 2. `or_csv_demo.py`
#### 3. `llm_to_or_csv_demo.py`
#### 4. `or_to_llm_csv_demo.py`

### 具体修改位置

在每个脚本的 `main()` 函数中，找到类似这样的代码：

```python
# 当前代码 (约在第509-512行)
unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
print(f"\nUsing unified initial samples for all items: {unified_samples}")
```

**需要替换为**:

```python
# 方案1: 从 train.csv 动态读取 (推荐)
test_path = Path(args.demand_file)
train_file = test_path.parent / "train.csv"

if train_file.exists():
    train_df = pd.read_csv(train_file)
    # Use weeks 1-9 from train.csv (exclude week 0 which is typically 0)
    train_samples = train_df[train_df['week_number'] >= 1]['demand'].tolist()
    initial_samples = {item_id: train_samples for item_id in csv_player.get_item_ids()}
    print(f"\nUsing initial samples from train.csv: {train_samples}")
    print(f"  Mean: {sum(train_samples)/len(train_samples):.1f}, Count: {len(train_samples)}")
else:
    # Fallback to default samples if train.csv not found
    unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
    initial_samples = {item_id: unified_samples.copy() for item_id in csv_player.get_item_ids()}
    print(f"\nTrain.csv not found, using default samples: {unified_samples}")
```

**需要添加的 import** (在文件顶部):
```python
from pathlib import Path  # 如果还没有导入
```

### 替代方案: 硬编码初始样本字典

如果不想每次都读取 train.csv，可以在脚本开头添加：

```python
# 在 import 语句之后，main() 函数之前
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
```

然后在 main() 中:
```python
# Get item_id from CSV
item_ids = csv_player.get_item_ids()
item_id = item_ids[0] if item_ids else None

# Use pre-defined samples if available
if item_id and item_id in REAL_INSTANCES_INITIAL_SAMPLES:
    train_samples = REAL_INSTANCES_INITIAL_SAMPLES[item_id]
    initial_samples = {item_id: train_samples}
    print(f"\nUsing pre-defined initial samples for {item_id}: {train_samples}")
else:
    # Fallback
    unified_samples = [108, 74, 119, 124, 51, 67, 103, 92, 100, 79]
    initial_samples = {item_id: unified_samples.copy() for item_id in item_ids}
    print(f"\nUsing default samples: {unified_samples}")
```

---

## 📋 测试检查清单

完成脚本修改后，请测试：

- [ ] 能否正确读取 test.csv 文件
- [ ] Item ID 是否正确识别 (例如 "1047675")
- [ ] 初始样本是否从 train.csv 正确加载
- [ ] lead_time 是否设置为 4
- [ ] profit 和 holding_cost 是否为 2 和 1
- [ ] News 列信息是否正确显示
- [ ] 游戏能否正常运行 50 周
- [ ] 最终奖励是否合理

---

## 🎯 下一步行动

1. **我可以帮你修改这4个测试脚本**，添加从 train.csv 读取初始样本的功能
2. **或者你可以选择使用哪种方案** (动态读取 vs 硬编码字典)
3. **测试一个完整示例**，确保所有功能正常工作

请告诉我你希望如何处理初始样本的读取！
