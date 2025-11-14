#!/usr/bin/env python3
"""
Test script for VMamba integration with RT-DETR.

This script validates that the VMamba backbone is correctly integrated
into the Ultralytics framework and can be used with RT-DETR.

Usage:
    python test_vmamba_integration.py
"""

import torch
from ultralytics import RTDETR
from ultralytics.nn.modules.vmamba import VisionMambaBackbone, VMambaStage
import sys


def test_vmamba_modules():
    """Test VMamba modules independently."""
    print("=" * 80)
    print("Testing VMamba Modules")
    print("=" * 80)

    # Test 1: VMambaStage
    print("\n1. Testing VMambaStage...")
    try:
        stage = VMambaStage(c1=96, c2=192, n=2)
        x = torch.randn(2, 96, 56, 56)  # Batch=2, C=96, H=56, W=56
        y = stage(x)
        print(f"   ✓ VMambaStage input shape: {x.shape}")
        print(f"   ✓ VMambaStage output shape: {y.shape}")
        assert y.shape == (2, 192, 56, 56), f"Expected (2, 192, 56, 56), got {y.shape}"
        print("   ✓ VMambaStage test PASSED")
    except Exception as e:
        print(f"   ✗ VMambaStage test FAILED: {e}")
        return False

    # Test 2: VisionMambaBackbone
    print("\n2. Testing VisionMambaBackbone...")
    try:
        backbone = VisionMambaBackbone(
            in_chans=3,
            depths=[2, 2, 5, 2],
            dims=[96, 192, 384, 768],
            out_indices=(1, 2, 3),  # Output from stages 1, 2, 3
        )
        x = torch.randn(2, 3, 224, 224)  # Standard input
        features = backbone(x)
        print(f"   ✓ VisionMambaBackbone input shape: {x.shape}")
        print(f"   ✓ VisionMambaBackbone output features: {len(features)} scales")
        for i, feat in enumerate(features):
            print(f"      - Feature {i}: {feat.shape}")
        assert len(features) == 3, f"Expected 3 output features, got {len(features)}"
        print("   ✓ VisionMambaBackbone test PASSED")
    except Exception as e:
        print(f"   ✗ VisionMambaBackbone test FAILED: {e}")
        return False

    return True


def test_rtdetr_vmamba_model():
    """Test RT-DETR with VMamba backbone."""
    print("\n" + "=" * 80)
    print("Testing RT-DETR with VMamba Backbone")
    print("=" * 80)

    config_files = [
        "ultralytics/cfg/models/rt-detr/rtdetr-vmamba-tiny.yaml",
        "ultralytics/cfg/models/rt-detr/rtdetr-vmamba-small.yaml",
    ]

    for config_file in config_files:
        print(f"\n3. Testing {config_file}...")
        try:
            # Create model from config
            model = RTDETR(config_file)
            print(f"   ✓ Model created successfully")

            # Test forward pass
            x = torch.randn(1, 3, 640, 640)  # Standard RTDETR input size
            print(f"   ✓ Input shape: {x.shape}")

            # Set model to eval mode for inference
            model.model.eval()
            with torch.no_grad():
                outputs = model.model(x)

            print(f"   ✓ Forward pass completed")
            print(f"   ✓ Output type: {type(outputs)}")

            if isinstance(outputs, (list, tuple)):
                print(f"   ✓ Number of outputs: {len(outputs)}")
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor):
                        print(f"      - Output {i} shape: {out.shape}")
            elif isinstance(outputs, torch.Tensor):
                print(f"   ✓ Output shape: {outputs.shape}")

            print(f"   ✓ {config_file} test PASSED")

            # Print model summary
            print(f"\n   Model Summary for {config_file.split('/')[-1]}:")
            model.model.info(verbose=False, imgsz=640)

        except Exception as e:
            print(f"   ✗ {config_file} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def test_model_parameters():
    """Test and compare model parameters."""
    print("\n" + "=" * 80)
    print("Model Parameters Comparison")
    print("=" * 80)

    configs = {
        "RT-DETR-L (Original)": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
        "RT-DETR-VMamba-Tiny": "ultralytics/cfg/models/rt-detr/rtdetr-vmamba-tiny.yaml",
        "RT-DETR-VMamba-Small": "ultralytics/cfg/models/rt-detr/rtdetr-vmamba-small.yaml",
    }

    print(f"\n{'Model':<30} {'Parameters':>15} {'GFLOPs':>10}")
    print("-" * 60)

    for name, config in configs.items():
        try:
            model = RTDETR(config)
            # Get model info
            info = model.model.info(verbose=False, imgsz=640)
            print(f"{name:<30} {'-':>15} {'-':>10}")
        except Exception as e:
            print(f"{name:<30} {'ERROR':>15} {'ERROR':>10}")
            print(f"   Error: {e}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VMamba Integration Test Suite")
    print("=" * 80)

    # Run tests
    tests = [
        ("VMamba Modules", test_vmamba_modules),
        ("RT-DETR-VMamba Models", test_rtdetr_vmamba_model),
        ("Model Parameters", test_model_parameters),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nTest '{test_name}' encountered an error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<40} {status}")

    # Overall result
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
