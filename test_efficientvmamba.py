#!/usr/bin/env python3
"""
Test script for EfficientVMamba RT-DETR backbone integration.

This script validates:
1. Module imports work correctly
2. Individual components can be instantiated
3. Forward pass produces correct output shapes
4. Full RT-DETR model can be created
"""

import torch
import sys
sys.path.insert(0, '.')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from ultralytics.nn.modules.efficientvmamba import (
            SS2D,
            VSSBlock,
            EfficientVMambaStem,
            EfficientVMambaBlock,
            EfficientVMambaStage,
            EfficientVMambaBackbone,
        )
        print("  ✓ All EfficientVMamba modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_ss2d():
    """Test SS2D (Selective Scan 2D) module."""
    print("Testing SS2D module...")
    try:
        from ultralytics.nn.modules.efficientvmamba import SS2D

        # Create module
        ss2d = SS2D(d_model=96, d_state=16, expand=2.0)

        # Test forward pass
        x = torch.randn(2, 16, 16, 96)  # (B, H, W, C)
        y = ss2d(x)

        assert y.shape == x.shape, f"Output shape {y.shape} != input shape {x.shape}"
        print(f"  ✓ SS2D: Input {x.shape} -> Output {y.shape}")
        return True
    except Exception as e:
        print(f"  ✗ SS2D test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vssblock():
    """Test VSSBlock (Visual State Space Block)."""
    print("Testing VSSBlock...")
    try:
        from ultralytics.nn.modules.efficientvmamba import VSSBlock

        block = VSSBlock(hidden_dim=96, d_state=16)
        x = torch.randn(2, 16, 16, 96)
        y = block(x)

        assert y.shape == x.shape, f"Output shape {y.shape} != input shape {x.shape}"
        print(f"  ✓ VSSBlock: Input {x.shape} -> Output {y.shape}")
        return True
    except Exception as e:
        print(f"  ✗ VSSBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stem():
    """Test EfficientVMambaStem."""
    print("Testing EfficientVMambaStem...")
    try:
        from ultralytics.nn.modules.efficientvmamba import EfficientVMambaStem

        stem = EfficientVMambaStem(c1=3, c2=96)
        x = torch.randn(2, 3, 640, 640)
        y = stem(x)

        expected_shape = (2, 96, 160, 160)  # stride 4
        assert y.shape == expected_shape, f"Output shape {y.shape} != expected {expected_shape}"
        print(f"  ✓ Stem: Input {x.shape} -> Output {y.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Stem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_block():
    """Test EfficientVMambaBlock."""
    print("Testing EfficientVMambaBlock...")
    try:
        from ultralytics.nn.modules.efficientvmamba import EfficientVMambaBlock

        block = EfficientVMambaBlock(dim=96, depth=2)
        x = torch.randn(2, 96, 160, 160)
        y = block(x)

        assert y.shape == x.shape, f"Output shape {y.shape} != input shape {x.shape}"
        print(f"  ✓ Block: Input {x.shape} -> Output {y.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Block test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage():
    """Test EfficientVMambaStage with downsampling."""
    print("Testing EfficientVMambaStage...")
    try:
        from ultralytics.nn.modules.efficientvmamba import EfficientVMambaStage

        stage = EfficientVMambaStage(c1=96, c2=192, depth=2, downsample=True)
        x = torch.randn(2, 96, 160, 160)
        y = stage(x)

        expected_shape = (2, 192, 80, 80)  # channels doubled, spatial halved
        assert y.shape == expected_shape, f"Output shape {y.shape} != expected {expected_shape}"
        print(f"  ✓ Stage: Input {x.shape} -> Output {y.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Stage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backbone():
    """Test complete EfficientVMamba backbone."""
    print("Testing EfficientVMambaBackbone...")
    try:
        from ultralytics.nn.modules.efficientvmamba import EfficientVMambaBackbone

        backbone = EfficientVMambaBackbone(variant="S", out_indices=(1, 2, 3))
        x = torch.randn(1, 3, 640, 640)
        outs = backbone(x)

        print(f"  Output features:")
        for i, out in enumerate(outs):
            print(f"    Stage {i+1}: {out.shape}")

        # Check expected shapes for variant S
        # Stage 1: 192 ch, stride 8 -> 80x80
        # Stage 2: 384 ch, stride 16 -> 40x40
        # Stage 3: 768 ch, stride 32 -> 20x20
        expected_shapes = [
            (1, 192, 80, 80),
            (1, 384, 40, 40),
            (1, 768, 20, 20),
        ]

        for out, expected in zip(outs, expected_shapes):
            assert out.shape == expected, f"Output shape {out.shape} != expected {expected}"

        print(f"  ✓ Backbone: All output shapes correct")
        return True
    except Exception as e:
        print(f"  ✗ Backbone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rtdetr_model():
    """Test full RT-DETR model with EfficientVMamba backbone."""
    print("Testing RT-DETR with EfficientVMamba backbone...")
    try:
        from ultralytics import RTDETR

        # Load model from YAML config
        model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-efficientvmamba.yaml")

        # Print model info
        print(f"  Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        x = torch.randn(1, 3, 640, 640)

        # Set to eval mode for inference
        model.eval()
        with torch.no_grad():
            results = model(x)

        print(f"  ✓ RT-DETR forward pass successful")
        return True
    except Exception as e:
        print(f"  ✗ RT-DETR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def count_parameters():
    """Count and compare parameters of different variants."""
    print("\nParameter count comparison:")
    try:
        from ultralytics.nn.modules.efficientvmamba import EfficientVMambaBackbone

        for variant in ["T", "S", "B"]:
            backbone = EfficientVMambaBackbone(variant=variant)
            params = sum(p.numel() for p in backbone.parameters())
            print(f"  EfficientVMamba-{variant}: {params:,} parameters")

        return True
    except Exception as e:
        print(f"  ✗ Parameter counting failed: {e}")
        return False


def main():
    print("=" * 60)
    print("EfficientVMamba RT-DETR Integration Test")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("SS2D Module", test_ss2d()))
    results.append(("VSSBlock", test_vssblock()))
    results.append(("Stem", test_stem()))
    results.append(("Block", test_block()))
    results.append(("Stage", test_stage()))
    results.append(("Backbone", test_backbone()))
    results.append(("Parameter Count", count_parameters()))
    results.append(("RT-DETR Model", test_rtdetr_model()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
