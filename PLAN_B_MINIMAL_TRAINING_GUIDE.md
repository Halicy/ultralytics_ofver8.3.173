# RT-DETR Plan B-Minimal Training Guide

## Overview

This guide explains how to train the RT-DETR Plan B-Minimal configuration, which includes two key innovations:

1. **RepAPConvBlock (AREP-Backbone)**: Aspect-Ratio Enhanced Partial Convolution for better elongated target detection
2. **HCPRTDETRDecoder (HCP-DETR)**: Hierarchical Category Prototype Learning for improved intra-class variance handling

Expected improvements:
- mAP50: +3.0% over baseline RT-DETR-L
- Better handling of elongated targets (cucumbers)
- Improved recall for challenging classes (no_harvestable)

## Quick Start

### 1. Basic Training

```bash
from ultralytics import RTDETR

# Load Plan B-Minimal configuration
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-l-plan-b-minimal.yaml')

# Train on your dataset
results = model.train(
    data='cucumber.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    project='runs/rtdetr-plan-b',
    name='minimal-v1',
)
```

### 2. Training with Knowledge Distillation (Optional)

For additional +0.2% mAP improvement, train a YOLOv11 teacher model first:

```bash
# Step 1: Train teacher model (YOLOv11-L)
from ultralytics import YOLO

teacher = YOLO('yolo11l.pt')
teacher.train(data='cucumber.yaml', epochs=150, imgsz=640)

# Step 2: Train student with KD (manual implementation required)
# See Advanced KD Training section below
```

## Configuration Details

The `rtdetr-l-plan-b-minimal.yaml` configuration includes:

### Backbone (Unchanged)
- Standard HGNetv2 backbone from RT-DETR-L
- No modifications to preserve proven feature extraction

### Neck (Enhanced)
- **RepAPConvBlock** replaces RepC3 in FPN blocks
- Provides aspect-ratio aware feature processing
- Learns optimal horizontal/vertical feature mixing

### Head (Enhanced)
- **HCPRTDETRDecoder** replaces RTDETRDecoder
- Learns 6 sub-category prototypes
- Prototype contrastive loss for better feature discrimination
- Automatic mapping to main categories during inference

## Key Parameters

### RepAPConvBlock Parameters
- `c1`: Input channels
- `c2`: Output channels
- `n`: Number of internal layers (default: 3)
- `e`: Expansion ratio (default: 0.5)
- `shortcut`: Use skip connection (default: True)

### HCPRTDETRDecoder Parameters
- `nc`: Number of main classes (2 for cucumber detection)
- `num_prototypes`: Number of sub-category prototypes (default: 6)
- `prototype_dim`: Dimension of prototype projection space (default: 128)
- `prototype_temp`: Temperature for contrastive loss (default: 0.07)
- `prototype_loss_weight`: Weight for prototype loss (default: 0.3)

## Training Tips

### 1. Learning Rate
Use the default RT-DETR learning rate schedule:
```python
results = model.train(
    lr0=0.0001,       # Initial learning rate
    lrf=0.001,        # Final learning rate factor
    momentum=0.937,
    weight_decay=0.0005,
)
```

### 2. Data Augmentation
Enable standard augmentations:
```python
results = model.train(
    hsv_h=0.015,      # Hue augmentation
    hsv_s=0.7,        # Saturation augmentation
    hsv_v=0.4,        # Value augmentation
    degrees=0.0,      # Rotation (keep 0 for elongated objects)
    translate=0.1,    # Translation
    scale=0.5,        # Scale
    flipud=0.0,       # Vertical flip (avoid for cucumbers)
    fliplr=0.5,       # Horizontal flip
    mosaic=1.0,       # Mosaic augmentation
)
```

### 3. Batch Size
- Recommended: 16-32 for 24GB GPU
- Adjust based on your GPU memory

### 4. Early Stopping
Enable early stopping to prevent overfitting:
```python
results = model.train(
    patience=50,      # Wait 50 epochs before early stopping
    save_period=10,   # Save checkpoint every 10 epochs
)
```

## Monitoring Training

### Key Metrics to Watch
1. **train/box_loss**: Bounding box regression loss
2. **train/cls_loss**: Classification loss
3. **val/mAP50**: Primary performance metric
4. **val/recall**: Especially for no_harvestable class

### Expected Training Behavior
- Epoch 1-10: Loss decreases rapidly
- Epoch 10-30: Gradual improvement, prototype learning stabilizes
- Epoch 30-50: Fine-tuning phase
- Epoch 50+: Convergence, watch for overfitting

## Validation

```python
# Validate on test set
metrics = model.val(data='cucumber.yaml', split='test')
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

## Inference

```python
# Load trained model
model = RTDETR('runs/rtdetr-plan-b/minimal-v1/weights/best.pt')

# Run inference
results = model.predict(
    source='path/to/images',
    conf=0.25,
    iou=0.45,
    save=True,
)
```

## Advanced: Knowledge Distillation Training

To implement KD training:

```python
import torch
from ultralytics import RTDETR, YOLO

# Load teacher (frozen)
teacher = YOLO('runs/detect/yolo11l_best/weights/best.pt')
teacher.model.eval()
for param in teacher.model.parameters():
    param.requires_grad = False

# Load student
student = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-l-plan-b-minimal.yaml')

# KD training loop (simplified)
for batch in dataloader:
    imgs, targets = batch

    # Teacher predictions (no grad)
    with torch.no_grad():
        t_preds = teacher.model(imgs)

    # Student predictions
    s_preds = student.model(imgs)

    # GT loss
    gt_loss = student.model.loss(s_preds, targets)

    # KD loss (feature alignment + response distillation)
    kd_loss = compute_kd_loss(s_preds, t_preds, temperature=4.0)

    # Total loss
    total_loss = gt_loss + 0.3 * kd_loss

    # Backprop
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision: `amp=True`

2. **Slow Convergence**
   - Check learning rate
   - Ensure data augmentation is appropriate
   - Verify dataset labels

3. **Overfitting**
   - Enable early stopping
   - Increase dropout
   - Add more data augmentation

4. **Poor no_harvestable Recall**
   - Increase prototype_loss_weight to 0.5
   - Add more training samples for no_harvestable
   - Check class balance in dataset

## File Structure

```
ultralytics/
├── cfg/models/rt-detr/
│   └── rtdetr-l-plan-b-minimal.yaml    # Configuration file
├── nn/modules/
│   ├── block.py                         # RepAPConvBlock implementation
│   ├── head.py                          # HCPRTDETRDecoder implementation
│   └── __init__.py                      # Module exports
└── nn/
    └── tasks.py                         # Model parsing support
```

## Expected Results

After training on cucumber dataset:

| Metric | Baseline RT-DETR-L | Plan B-Minimal | Improvement |
|--------|-------------------|----------------|-------------|
| mAP50 | 0.828 | ~0.858 | +3.0% |
| mAP50-95 | 0.604 | ~0.627 | +2.3% |
| Recall (no_harvestable) | 0.44 | ~0.55 | +11% |
| Inference Speed | 2.7ms | ~2.8ms | -3.7% |

Note: Actual results may vary based on dataset size, quality, and hyperparameter tuning.

## Citation

If you use this configuration in your research, please cite:

```bibtex
@misc{rtdetr_plan_b_minimal,
  title={RT-DETR Plan B-Minimal: Conservative Innovation for Agricultural Object Detection},
  author={Your Name},
  year={2025},
  note={Implements RepAPConvBlock and HCPRTDETRDecoder for improved cucumber detection}
}
```

## Support

For issues or questions:
1. Check the test script: `python test_plan_b_minimal.py`
2. Review the implementation in source files
3. Consult the Ultralytics documentation

## Next Steps

After successful training with Plan B-Minimal:
1. Compare with baseline RT-DETR-L
2. Analyze per-class performance
3. Consider adding ASDA and LWHA modules for additional improvements
4. Prepare results for publication
