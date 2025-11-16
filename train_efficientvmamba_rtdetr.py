#!/usr/bin/env python3
"""
Training script for RT-DETR with EfficientVMamba backbone.

This script demonstrates how to train RT-DETR with the integrated
EfficientVMamba backbone on a custom dataset.

Usage:
    python train_efficientvmamba_rtdetr.py --data path/to/data.yaml --variant S
"""

import argparse
from pathlib import Path


def train_efficientvmamba_rtdetr(
    data_yaml: str,
    variant: str = "S",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/efficientvmamba_rtdetr",
    name: str = "exp",
    pretrained: bool = False,
    resume: bool = False,
    workers: int = 8,
):
    """
    Train RT-DETR with EfficientVMamba backbone.

    Args:
        data_yaml: Path to dataset YAML file
        variant: EfficientVMamba variant (T, S, or B)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: CUDA device
        project: Project directory for saving results
        name: Experiment name
        pretrained: Use pretrained weights
        resume: Resume training
        workers: Number of data loading workers
    """
    from ultralytics import RTDETR

    # Select model configuration based on variant
    variant_map = {
        "T": "ultralytics/cfg/models/rt-detr/rtdetr-efficientvmamba-t.yaml",
        "S": "ultralytics/cfg/models/rt-detr/rtdetr-efficientvmamba.yaml",
        "B": "ultralytics/cfg/models/rt-detr/rtdetr-efficientvmamba-b.yaml",
    }

    if variant.upper() not in variant_map:
        raise ValueError(f"Invalid variant: {variant}. Choose from T, S, or B")

    model_yaml = variant_map[variant.upper()]

    print(f"=" * 60)
    print(f"RT-DETR with EfficientVMamba-{variant.upper()} Training")
    print(f"=" * 60)
    print(f"Model Config: {model_yaml}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Device: {device}")
    print(f"=" * 60)

    # Load model
    if resume:
        model = RTDETR(f"{project}/{name}/weights/last.pt")
    else:
        model = RTDETR(model_yaml)

    # Print model summary
    print("\nModel Architecture:")
    print(model.info())

    # Training configuration
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "workers": workers,
        "patience": 50,
        "save": True,
        "save_period": 10,
        "cache": False,
        "pretrained": pretrained,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": True,
        "close_mosaic": 10,
        "resume": resume,
    }

    # Start training
    print("\nStarting training...")
    results = model.train(**train_args)

    print("\nTraining completed!")
    print(f"Results saved to: {project}/{name}")

    return results


def validate_model(
    weights_path: str,
    data_yaml: str,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
):
    """
    Validate trained model.

    Args:
        weights_path: Path to trained weights
        data_yaml: Path to dataset YAML file
        imgsz: Input image size
        batch: Batch size
        device: CUDA device
    """
    from ultralytics import RTDETR

    print(f"Validating model: {weights_path}")

    model = RTDETR(weights_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    print("\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return metrics


def export_model(
    weights_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
):
    """
    Export trained model.

    Args:
        weights_path: Path to trained weights
        format: Export format (onnx, torchscript, etc.)
        imgsz: Input image size
        half: Use FP16
        simplify: Simplify ONNX model
    """
    from ultralytics import RTDETR

    print(f"Exporting model: {weights_path}")
    print(f"Format: {format}")

    model = RTDETR(weights_path)
    model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        simplify=simplify,
    )

    print("Export completed!")


def create_sample_dataset_yaml():
    """Create a sample dataset YAML configuration."""
    sample_yaml = """
# Sample dataset configuration for RT-DETR training
# Replace paths with your actual dataset paths

# Path to dataset root directory
path: /path/to/dataset

# Train/val/test image directories (relative to 'path')
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 80

# Class names
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  # ... add all your classes
  # For COCO, you would have 80 classes

# Optional: Download script
# download: https://example.com/dataset.zip
"""

    print("Sample dataset YAML configuration:")
    print(sample_yaml)
    print("\nSave this to a file (e.g., data.yaml) and update paths accordingly.")


def main():
    parser = argparse.ArgumentParser(
        description="Train RT-DETR with EfficientVMamba backbone"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="S",
        choices=["T", "S", "B"],
        help="EfficientVMamba variant (T=Tiny, S=Small, B=Base)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device (e.g., 0 or 0,1,2,3 or cpu)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/efficientvmamba_rtdetr",
        help="Project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate model weights (provide path to weights)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export model weights (provide path to weights)",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="onnx",
        help="Export format",
    )
    parser.add_argument(
        "--sample-yaml",
        action="store_true",
        help="Print sample dataset YAML configuration",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print model information only",
    )

    args = parser.parse_args()

    if args.sample_yaml:
        create_sample_dataset_yaml()
        return

    if args.info:
        from ultralytics import RTDETR

        variant_map = {
            "T": "ultralytics/cfg/models/rt-detr/rtdetr-efficientvmamba-t.yaml",
            "S": "ultralytics/cfg/models/rt-detr/rtdetr-efficientvmamba.yaml",
            "B": "ultralytics/cfg/models/rt-detr/rtdetr-efficientvmamba-b.yaml",
        }
        model_yaml = variant_map[args.variant.upper()]
        model = RTDETR(model_yaml)
        print(model.info(detailed=True))
        return

    if args.validate:
        if not args.data:
            raise ValueError("--data is required for validation")
        validate_model(
            weights_path=args.validate,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        return

    if args.export:
        export_model(
            weights_path=args.export,
            format=args.export_format,
            imgsz=args.imgsz,
        )
        return

    if not args.data:
        print("Error: --data is required for training")
        print("Use --sample-yaml to see a sample dataset configuration")
        parser.print_help()
        return

    train_efficientvmamba_rtdetr(
        data_yaml=args.data,
        variant=args.variant,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
