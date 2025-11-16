
from ultralytics import RTDETR
import torch


def main():
    # 1. 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 创建 RT-DETR-VMamba-Tiny 模型
    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-vmamba-tiny.yaml")

    # 3. 配置训练参数
    train_args = {
        "data": "cucumber_dataset.yaml",  # 数据集配置（已在项目根目录）
        "epochs": 120,
        "imgsz": 640,
        "batch": 8,
        "device": device,

        # 优化器
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.05,

        # 数据增强
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,

        # 训练策略
        "patience": 50,
        "save": True,
        "save_period": 10,
        "val": True,

        # 项目目录
        "project": "runs/rtdetr-vmamba",
        "name": "cucumber_detection_tiny",
        "exist_ok": False,
        "pretrained": False,
        "verbose": True,
    }

    # 4. 启动训练
    results = model.train(**train_args)

    # 5. 训练后验证
    metrics = model.val()
    print(f"Final mAP50-95: {metrics.box.map:.4f}")
    print(f"Final mAP50: {metrics.box.map50:.4f}")
    print(f"Final mAP75: {metrics.box.map75:.4f}")


if __name__ == "__main__":
    main()
