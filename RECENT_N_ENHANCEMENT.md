# Recent-N Enhancement for LLM→OR Strategy

## 概述

为 `llm_to_or_csv_demo.py` 添加了 `Recent-N` 方法到 lead time (L) 参数，并强化了所有三个参数 (L, mu_hat, sigma_hat) 的 `Recent-N` 选择策略。

## 核心改进

### 1. 为 L 添加 Recent-N 方法

**修改内容：**

**A. `compute_L()` 函数：**
```python
def compute_L(method: str, params: dict, observed_lead_times: List[int], promised_lead_time: float) -> float:
    # 新增 "recent_N" 方法支持
    if method == "recent_N":
        if not observed_lead_times:
            raise ValueError("Cannot compute recent_N lead time: no observed arrivals yet")
        if "N" not in params:
            raise ValueError("Method 'recent_N' for L requires 'N' field")
        N = int(params["N"])
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        recent_samples = observed_lead_times[-N:] if len(observed_lead_times) >= N else observed_lead_times
        return float(np.mean(recent_samples))
```

**B. 更新方法支持列表：**
- `default`: 使用承诺的 lead time
- `calculate`: 使用所有观测到的 lead time 的平均值
- **`recent_N`**: 使用最近 N 个观测到的 lead time 的平均值 ✨新增
- `explicit`: 手动指定值

### 2. 强化 Recent-N 的 N 值选择策略

**设计理念：**
- **核心动机**: 如果某参数发生明显变化，更应使用近期的均值/标准差作为预测值
- **独立性**: L, mu_hat, sigma_hat 可能有不同的 changepoint，因此使用不同的 N 值
- **自适应性**: N 值基于 changepoint 检测动态计算

**四步策略：**

#### Step 1: 检测 Changepoint
使用简单的启发式规则：

**对于 demand 参数 (mu_hat, sigma_hat):**
- Mean shift: 持续 3+ 天的 30%+ 变化
- Variance shift: 方差加倍或减半
- News impact: 有持久影响的新闻事件
- Trend reversal: 需求模式的趋势反转

**对于 lead time 参数 (L):**
- 在 arrival records 中检测持续的变化
- 例如："从 Day X 开始，lead_time 从 2 天变为 4 天"

#### Step 2: 计算 Regime Length
使用公式：
```
N = (current_day - changepoint_day) + 1
```

**示例：**
- Changepoint: Day 15, Current: Day 20 → regime_length = 6 → N = 6
- Changepoint: Day 10, Current: Day 25 → regime_length = 16 → N = 16

#### Step 3: 应用自适应约束
```
if regime_length < 3:
    N = 3                    # 最小值
elif regime_length > 20:
    N = 20                   # 最大值
elif samples_available < N:
    N = samples_available    # 可用样本数限制
else:
    N = regime_length        # 计算值
```

**示例：**
- 计算 N = 1 → 约束为 N = 3
- 计算 N = 30 → 约束为 N = 20
- 无 changepoint → N = 10 (默认值，平衡稳定期)

#### Step 4: 在 rationale 中明确说明
必须明确说明：
- 检测到的 changepoint 及其证据
- 计算的 N 值及原因

### 3. Prompt 修改

**A. L 的方法说明：**
```
1. L (lead time for current order):
   - default: Use the supplier-promised lead time shown above
   - calculate: Use average of all observed lead times from past arrivals
   - recent_N: Use average of last N observed lead times (must specify N) ✨新增
   - explicit: Provide your own predicted value
   Example: {"method": "calculate"} or {"method": "recent_N", "N": 5} or {"method": "explicit", "value": 3}
```

**B. 统一的 Recent-N 策略：**
```
⚠️ IMPORTANT: When choosing recent_N for L, mu_hat, or sigma_hat:
The three parameters may have DIFFERENT change-points and thus DIFFERENT N values!

STRATEGY for setting N when using recent_N:
Step 1: Detect the most recent change-point for THIS parameter using simple heuristics:
        • For demand (mu_hat/sigma_hat): Look for mean/variance shifts (>30% sustained over 3+ days),
          news events with lasting impact, or trend reversals in demand patterns.
        • For lead time (L): Look for sustained lead_time changes in arrival records
          (e.g., shifted from 2 to 4 days starting Day X).

Step 2: Calculate regime length using the formula:
        N = (current_day - changepoint_day) + 1

Step 3: Apply adaptive constraints:
        • N = 3 (minimum) if regime_length < 3
        • N = 20 (maximum) if regime_length > 20
        • N = sample_count if fewer samples than calculated N
        • Otherwise: N = regime_length

Step 4: In your rationale, explicitly state:
        • Which changepoint you detected and the evidence
        • The calculated N value and why

Examples of N calculation:
  • Detected demand change at Day 15, current Day 20: regime_length = 6 → N = 6
  • Detected lead_time change at Day 10, current Day 25: regime_length = 16 → N = 16
  • Change at Day 5, current Day 5: regime_length = 1 → N = 3 (applied minimum)
  • Change at Day 1, current Day 30: regime_length = 30 → N = 20 (applied maximum)
  • No clear change detected: Use N = 10 as default (balanced for stable periods)
```

**C. 输出格式说明：**
```
IMPORTANT:
- For L: 'recent_N' requires 'N', 'explicit' requires 'value', others require no extra field
- For mu_hat: 'recent_N' requires 'N', 'EWMA_gamma' requires 'gamma', 'explicit' requires 'value'
- For sigma_hat: 'recent_N' requires 'N', 'explicit' requires 'value', others require no extra field
- All 'N' values are integers >= 1 and should be chosen based on changepoint detection
- All 'value' fields are numeric
- Do NOT include any text outside the JSON
- Think carefully about your parameter choices and changepoint reasoning
```

### 4. 验证和日志

**验证逻辑：**
```python
# 更新了 validate_parameters_json() 函数
L_method = L_param["method"]
if L_method not in ["default", "calculate", "recent_N", "explicit"]:
    raise ValueError(f"Invalid L method: {L_method}")
if L_method == "recent_N" and "N" not in L_param:
    raise ValueError("Method 'recent_N' for L requires 'N' field")
```

**日志输出：**
```python
# 在 main() 中更新了日志显示
l_method = item_params['L']['method']
l_extra = ""
if l_method == 'explicit' and 'value' in item_params['L']:
    l_extra = f", value={item_params['L']['value']}"
elif l_method == 'recent_N' and 'N' in item_params['L']:  # ✨新增
    l_extra = f", N={int(item_params['L']['N'])}"
elif l_method == 'calculate':
    l_extra = f", observed_samples={len(observed_lead_times[item_id])}"
print(f"  L method: {l_method}{l_extra}, computed L = {L:.2f}")
```

## 使用示例

### 场景 1: Lead Time 变化检测

**LLM 检测到的 Changepoint:**
```
Day 10-13 arrivals show lead_time=4 days consistently (was 2 days before)
Changepoint detected at: Day 10
```

**LLM 的计算:**
```
current_day = 20
changepoint = 10
regime_length = (20 - 10) + 1 = 11
N = 11 (within constraints 3-20)
```

**LLM 的输出:**
```json
{
  "rationale": "Detected lead_time change at Day 10 (shifted from 2 to 4 days in arrivals Days 10-13). Using recent_N with N=11 to reflect new regime.",
  "parameters": {
    "item_id": {
      "L": {"method": "recent_N", "N": 11},
      ...
    }
  }
}
```

**后端计算:**
```
observed_lead_times = [2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
recent_samples = observed_lead_times[-11:] = [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
L = mean([2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]) = 3.82
```

### 场景 2: 需求方差变化

**LLM 检测到的 Changepoint:**
```
Days 1-14: mean=100, std=25 (stable)
Days 15-20: mean=100, std=50 (variance doubled)
Changepoint detected at: Day 15 for sigma_hat
```

**LLM 的计算:**
```
current_day = 20
changepoint = 15
regime_length = (20 - 15) + 1 = 6
N = 6 (within constraints 3-20)
```

**LLM 的输出:**
```json
{
  "parameters": {
    "item_id": {
      "L": {"method": "calculate"},
      "mu_hat": {"method": "default"},  // Mean unchanged
      "sigma_hat": {"method": "recent_N", "N": 6}  // Only variance changed
    }
  }
}
```

**特点**: L 和 mu_hat 使用所有历史数据，只有 sigma_hat 使用 recent_N！

### 场景 3: 边界情况

**A. Regime Length < 3:**
```
Changepoint: Day 20, Current: Day 20
Calculated: regime_length = 1
Applied: N = 3 (minimum constraint)
```

**B. Regime Length > 20:**
```
Changepoint: Day 1, Current: Day 30
Calculated: regime_length = 30
Applied: N = 20 (maximum constraint)
```

**C. No Changepoint Detected:**
```
No clear changepoint
Applied: N = 10 (default for stable periods)
```

### 场景 4: 三个参数独立选择

**示例输出:**
```json
{
  "rationale": "L: No change in lead_time (using calculate). mu_hat: Demand increased 50% at Day 5 (using recent_N with N=16). sigma_hat: Variance doubled at Day 15 (using recent_N with N=6).",
  "parameters": {
    "item_id": {
      "L": {"method": "calculate"},                    // N/A
      "mu_hat": {"method": "recent_N", "N": 16},       // 独立 N
      "sigma_hat": {"method": "recent_N", "N": 6}      // 独立 N
    }
  }
}
```

**关键点：**
- ✅ L: 15 个观测样本，均值 = 2.1
- ✅ mu_hat: 最近 16 天的平均需求
- ✅ sigma_hat: 最近 6 天的标准差
- ✅ 三个参数使用了**不同的 N 值**！

## 效果预期

### 改进前 vs 改进后

| 方面 | 改进前 | 改进后 |
|------|--------|--------|
| **L 的方法** | default, calculate, explicit | + recent_N ✨ |
| **N 的选择** | 笼统的规则，容易用固定值 | 明确的 4 步策略 + 数学公式 |
| **Changepoint 检测** | 未强调 | 简单启发式规则 |
| **参数独立性** | 未强调 | 明确允许不同 N 值 |
| **约束机制** | 未提供 | 自适应 min/max 约束 |
| **可审计性** | rationale 要求弱 | 明确要求说明证据 |

### 对 LLM 的影响

**积极方面：**
1. ✅ **更灵活**: L 现在也能适应 lead time 的变化
2. ✅ **更智能**: N 值基于 changepoint，不是固定值
3. ✅ **更精确**: 不同参数使用不同的时间窗口
4. ✅ **更可审计**: rationale 中明确记录决策过程

**示例场景：**
- **Before**: L=2 (fixed promise), 即使 lead_time 已经变成 4 天
- **After**: L=4 (recent_N with appropriate N), 反映真实的当前状态

## 代码一致性

### 对比：mu_hat 和 sigma_hat 的 recent_N

```python
# mu_hat 的 recent_N
def compute_mu_hat(method, params, samples, L):
    if method == "recent_N":
        N = int(params["N"])
        recent_samples = samples[-N:] if len(samples) >= N else samples
        empirical_mean = np.mean(recent_samples)
        return (1 + L) * empirical_mean

# sigma_hat 的 recent_N
def compute_sigma_hat(method, params, samples, L):
    if method == "recent_N":
        N = int(params["N"])
        recent_samples = samples[-N:] if len(samples) >= N else samples
        empirical_std = np.std(recent_samples, ddof=1)
        return np.sqrt(1 + L) * empirical_std

# L 的 recent_N (新增)
def compute_L(method, params, observed_lead_times, promised_lead_time):
    if method == "recent_N":
        N = int(params["N"])
        recent_samples = observed_lead_times[-N:] if len(observed_lead_times) >= N else observed_lead_times
        return float(np.mean(recent_samples))
```

**结构完全一致！** ✨

## 总结

✅ **新增功能:**
1. L 支持 `recent_N` 方法
2. 统一的 4 步 N 值选择策略
3. 自适应的 min/max 约束 (3-20)
4. 强化 prompt 指导

✅ **核心优势:**
1. 更准确反映当前状态（特别是 L）
2. 防止过度依赖历史数据
3. 参数独立适配各自 changepoint
4. 更可审计和可解释

✅ **设计原则:**
1. **简单性**: 不引入过于复杂的 changepoint 检测算法
2. **一致性**: L, mu_hat, sigma_hat 的 recent_N 逻辑一致
3. **自适应**: 根据实际 changepoint 动态计算 N
4. **鲁棒性**: 边界情况有明确的处理

这次增强使 `llm_to_or_csv_demo.py` 能够更好地适应时变的 lead time 和需求模式！🎯

