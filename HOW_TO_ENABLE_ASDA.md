# 🔧 如何启用 ASDA - 一步到位指南

## 📍 只需修改 1 行代码！

要启用 ASDA（长宽比感知的可变形注意力），你只需要修改 **1 行代码**。

---

## ⚡ 方法 1: 手动修改（推荐）

### 步骤 1: 打开文件

```bash
# 在你的项目路径下
cd /root/autodl-tmp/projects/ultralytics

# 编辑 transformer.py
vim ultralytics/nn/modules/transformer.py +639
# 或使用你喜欢的编辑器
```

### 步骤 2: 找到第 639 行

找到 `DeformableTransformerDecoderLayer` 类的 `__init__` 方法中的这一行：

```python
# Cross attention
self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
```

### 步骤 3: 修改为

```python
# Cross attention with ASDA (Aspect-ratio Sensitive Deformable Attention)
self.cross_attn = ASDA(d_model, n_levels, n_heads, n_points, aspect_ratio_range=(1.0, 10.0))
```

### 步骤 4: 保存并退出

就这样！ASDA 已启用 ✅

---

## ⚡ 方法 2: 使用 sed 命令（快速）

```bash
cd /root/autodl-tmp/projects/ultralytics

# 备份原文件
cp ultralytics/nn/modules/transformer.py ultralytics/nn/modules/transformer.py.bak

# 自动替换
sed -i 's/self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)/self.cross_attn = ASDA(d_model, n_levels, n_heads, n_points, aspect_ratio_range=(1.0, 10.0))/g' ultralytics/nn/modules/transformer.py

# 验证修改
grep -n "self.cross_attn = ASDA" ultralytics/nn/modules/transformer.py
```

**预期输出**:
```
639:        self.cross_attn = ASDA(d_model, n_levels, n_heads, n_points, aspect_ratio_range=(1.0, 10.0))
```

---

## ⚡ 方法 3: 使用 Python 脚本（自动化）

创建并运行这个脚本：

```python
# enable_asda.py
import os

transformer_path = "/root/autodl-tmp/projects/ultralytics/ultralytics/nn/modules/transformer.py"

# 读取文件
with open(transformer_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换
old_line = "self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)"
new_line = "self.cross_attn = ASDA(d_model, n_levels, n_heads, n_points, aspect_ratio_range=(1.0, 10.0))"

if old_line in content:
    content = content.replace(old_line, new_line)

    # 写回文件
    with open(transformer_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ ASDA 已成功启用！")
    print(f"修改位置: {transformer_path}:639")
else:
    print("⚠️  未找到需要替换的代码，可能已经修改过或文件结构不同")
```

运行：
```bash
python enable_asda.py
```

---

## 🔍 验证 ASDA 是否启用

### 方法 1: 代码验证

```python
from ultralytics import RTDETR

# 加载模型
model = RTDETR('rtdetr-l.yaml')

# 检查 decoder layer
decoder_layer = model.model.model[-1].decoder.layers[0]
cross_attn_type = type(decoder_layer.cross_attn).__name__

print(f"Cross Attention Type: {cross_attn_type}")

if cross_attn_type == "ASDA":
    print("✅ ASDA 已启用！")
    print(f"   长宽比范围: {decoder_layer.cross_attn.aspect_ratio_range}")
else:
    print(f"❌ 未启用 ASDA，当前使用: {cross_attn_type}")
```

### 方法 2: 训练时日志

训练时，模型摘要中会显示 ASDA：

```
Model summary: XXX layers, XXX parameters, XXX gradients, XXX GFLOPs
...
  23                -1  1      XXXX  ultralytics.nn.modules.transformer.ASDA
...
```

---

## 📋 完整修改前后对比

### 修改前（使用标准 MSDeformAttn）

```python
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)  # ← 原始
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        ...
```

### 修改后（使用 ASDA）

```python
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention with ASDA
        self.cross_attn = ASDA(d_model, n_levels, n_heads, n_points, aspect_ratio_range=(1.0, 10.0))  # ← 修改
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        ...
```

---

## 🎛️ ASDA 参数调整

### aspect_ratio_range 说明

这个参数控制 ASDA 预测的长宽比范围：

```python
aspect_ratio_range=(min_ratio, max_ratio)
```

#### 推荐设置

| 目标类型 | 长宽比范围 | 示例 |
|---------|----------|------|
| **极细长目标** | `(1.0, 10.0)` | 黄瓜、胡萝卜、豆角 |
| **中等细长** | `(1.0, 5.0)` | 香蕉、茄子 |
| **略细长** | `(1.0, 3.0)` | 番茄、苹果（椭圆形） |
| **近方形** | `(1.0, 2.0)` | 南瓜、西瓜 |

#### 示例

```python
# 对于黄瓜（长宽比通常 5:1 到 10:1）
self.cross_attn = ASDA(..., aspect_ratio_range=(1.0, 10.0))

# 对于香蕉（长宽比通常 3:1 到 5:1）
self.cross_attn = ASDA(..., aspect_ratio_range=(1.0, 5.0))

# 对于番茄（长宽比通常 1:1 到 1.5:1）
self.cross_attn = ASDA(..., aspect_ratio_range=(1.0, 2.0))
```

---

## 🚨 常见错误

### 错误 1: NameError: name 'ASDA' is not defined

**原因**: ASDA 没有导入

**解决**: 确保 `transformer.py` 文件中：
1. `__all__` 包含 `"ASDA"`（第 26 行）
2. ASDA 类存在（第 806-994 行）

### 错误 2: 修改后没有效果

**原因**: 使用了缓存的模型权重

**解决**:
```bash
# 删除之前训练的权重（如果有）
rm -rf runs/detect/train*

# 重新训练
yolo detect train model=rtdetr-l.yaml ...
```

### 错误 3: 训练时 OOM (内存溢出)

**原因**: ASDA 略微增加了显存占用

**解决**:
```bash
# 减小 batch size
yolo detect train model=rtdetr-l.yaml batch=12  # 从 16 降到 12

# 或使用梯度累积
yolo detect train model=rtdetr-l.yaml batch=8 accumulate=2
```

---

## ✅ 修改检查清单

完成修改后，逐项检查：

- [ ] `ultralytics/nn/modules/transformer.py` 第 639 行已修改
- [ ] 使用 `grep` 命令确认修改：
  ```bash
  grep -n "self.cross_attn = ASDA" ultralytics/nn/modules/transformer.py
  ```
- [ ] 删除旧的训练权重（避免缓存）
- [ ] 准备开始新的训练

---

## 🎯 训练命令（启用 ASDA）

```bash
cd /root/autodl-tmp/projects/ultralytics

# 基础训练
yolo detect train \
    model=rtdetr-l.yaml \
    data=cucumber.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    lr0=0.0001 \
    project=runs/rtdetr-asda \
    name=exp1

# 对比 Baseline（未启用 ASDA）
# 1. 先恢复原文件: cp ultralytics/nn/modules/transformer.py.bak ultralytics/nn/modules/transformer.py
# 2. 训练 Baseline
# 3. 再启用 ASDA 训练
# 4. 对比结果
```

---

## 📊 预期训练日志

启用 ASDA 后，训练日志中会出现：

```
from ultralytics.nn.modules.transformer import ASDA
...
Model Summary: 303 layers, 32,527,850 parameters, ...
...
  DeformableTransformerDecoderLayer.cross_attn: ASDA
    - aspect_ratio_predictor: 16,640 params
    - ellipse_bias: 256 params
...
```

---

## 🎉 完成！

完成上述修改后，你的 RT-DETR 模型就已经集成了 **ASDA（长宽比感知的可变形注意力）**！

接下来：
1. 🏃 开始训练
2. 📊 对比 mAP 和召回率
3. 📈 分析混淆矩阵（关注 no_harvestable 类的改善）
4. 🎨 可视化长宽比预测

祝训练顺利！🚀
