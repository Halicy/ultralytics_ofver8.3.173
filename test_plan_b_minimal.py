#!/usr/bin/env python3
"""
Comprehensive Test Script for RT-DETR Plan B-Minimal Configuration
===================================================================

Tests the minimal fix configuration with:
  1. RepAPConvBlock (AREP-Backbone)
  2. HCPRTDETRDecoder (HCP-DETR with all bug fixes)
  3. Knowledge Distillation (via training script)

This script performs rigorous validation to ensure:
  - Module imports work correctly
  - Model instantiation succeeds
  - Forward pass works in eval and train modes
  - Backward pass works (gradients flow correctly)
  - No shape mismatches or type errors

Date: 2025-11-16
"""

import sys
import torch
import torch.nn as nn

# Test counters
passed = 0
failed = 0
total = 0


def test(name, condition, error_msg=""):
    """Run a single test."""
    global passed, failed, total
    total += 1
    if condition:
        print(f"✅ PASS: {name}")
        passed += 1
        return True
    else:
        print(f"❌ FAIL: {name}")
        if error_msg:
            print(f"   Error: {error_msg}")
        failed += 1
        return False


def test_module_imports():
    """Test that all innovation modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)

    try:
        from ultralytics.nn.modules import RepAPConvBlock, HCPRTDETRDecoder
        test("Import RepAPConvBlock", True)
        test("Import HCPRTDETRDecoder", True)
        return True
    except ImportError as e:
        test("Import modules", False, str(e))
        return False


def test_repapconvblock():
    """Test RepAPConvBlock instantiation and forward pass."""
    print("\n" + "="*70)
    print("TEST 2: RepAPConvBlock Module")
    print("="*70)

    try:
        from ultralytics.nn.modules import RepAPConvBlock

        # Test instantiation
        block = RepAPConvBlock(c1=256, c2=256, n=3, e=0.5, shortcut=True)
        test("RepAPConvBlock instantiation", True)

        # Test forward pass
        x = torch.randn(2, 256, 40, 40)
        y = block(x)
        test("RepAPConvBlock forward pass", y.shape == (2, 256, 40, 40),
             f"Expected (2, 256, 40, 40), got {y.shape}")

        # Test gradient flow
        loss = y.sum()
        loss.backward()
        has_grad = block.alpha.grad is not None
        test("RepAPConvBlock gradient flow", has_grad, "No gradient for alpha parameter")

        # Test aspect ratio weight
        weight = block.get_aspect_ratio_weight()
        test("RepAPConvBlock aspect ratio weight", 0.0 <= weight <= 1.0,
             f"Weight {weight} out of range [0, 1]")

        return True
    except Exception as e:
        test("RepAPConvBlock", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_hcprtdetrdecoder():
    """Test HCPRTDETRDecoder instantiation and forward pass."""
    print("\n" + "="*70)
    print("TEST 3: HCPRTDETRDecoder Module")
    print("="*70)

    try:
        from ultralytics.nn.modules import HCPRTDETRDecoder

        # Test instantiation
        decoder = HCPRTDETRDecoder(
            nc=2,
            ch=(256, 256, 256),
            hd=256,
            nq=300,
            ndp=4,
            nh=8,
            ndl=6,
            d_ffn=1024,
            dropout=0.0,
            act=nn.ReLU(),
            eval_idx=-1,
            nd=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False,
            num_prototypes=6,
            prototype_dim=128,
            prototype_temp=0.07,
            prototype_loss_weight=0.3,
        )
        test("HCPRTDETRDecoder instantiation", True)

        # Check prototypes
        test("HCPRTDETRDecoder has prototypes", hasattr(decoder, 'prototypes'),
             "Missing prototypes parameter")
        test("HCPRTDETRDecoder prototype shape", decoder.prototypes.shape == (6, 256),
             f"Expected (6, 256), got {decoder.prototypes.shape}")

        # Check projection heads
        test("HCPRTDETRDecoder has prototype_proj", hasattr(decoder, 'prototype_proj'),
             "Missing prototype projection head")
        test("HCPRTDETRDecoder has prototype_proj_p", hasattr(decoder, 'prototype_proj_p'),
             "Missing prototype projection for prototypes")

        # Check sub-to-main mapping
        test("HCPRTDETRDecoder has sub_to_main_map", hasattr(decoder, 'sub_to_main_map'),
             "Missing sub_to_main_map buffer")
        test("HCPRTDETRDecoder mapping shape", decoder.sub_to_main_map.shape == (6, 2),
             f"Expected (6, 2), got {decoder.sub_to_main_map.shape}")

        # Test forward pass in eval mode
        decoder.eval()
        x = [
            torch.randn(1, 256, 80, 80),  # P3
            torch.randn(1, 256, 40, 40),  # P4
            torch.randn(1, 256, 20, 20),  # P5
        ]
        with torch.no_grad():
            output = decoder(x)

        if isinstance(output, tuple):
            y = output[0]
        else:
            y = output

        test("HCPRTDETRDecoder eval forward pass", y.shape == (1, 300, 6),
             f"Expected (1, 300, 6), got {y.shape}")

        return True
    except Exception as e:
        test("HCPRTDETRDecoder", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_model_config():
    """Test loading the Plan B-Minimal YAML configuration."""
    print("\n" + "="*70)
    print("TEST 4: Model Configuration Loading")
    print("="*70)

    try:
        from ultralytics import RTDETR

        config_path = "ultralytics/cfg/models/rt-detr/rtdetr-l-plan-b-minimal.yaml"
        model = RTDETR(config_path)
        test("Load Plan B-Minimal config", True)

        # Check model has the right structure
        test("Model has model attribute", hasattr(model, 'model'),
             "Missing model attribute")

        # Check that HCPRTDETRDecoder is in the model
        has_hcp = False
        has_repap = False
        for name, module in model.model.named_modules():
            if "HCPRTDETRDecoder" in str(type(module)):
                has_hcp = True
            if "RepAPConvBlock" in str(type(module)):
                has_repap = True

        test("Model contains HCPRTDETRDecoder", has_hcp,
             "HCPRTDETRDecoder not found in model")
        test("Model contains RepAPConvBlock", has_repap,
             "RepAPConvBlock not found in model")

        return model
    except Exception as e:
        test("Load model config", False, str(e))
        import traceback
        traceback.print_exc()
        return None


def test_model_forward(model):
    """Test forward pass of the complete model."""
    print("\n" + "="*70)
    print("TEST 5: Model Forward Pass")
    print("="*70)

    if model is None:
        test("Model forward pass", False, "Model not loaded")
        return False

    try:
        # Test eval mode forward pass
        model.model.eval()
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            output = model.model(x)

        test("Model eval forward pass", output is not None, "Output is None")

        # Check output shape
        if isinstance(output, tuple):
            y = output[0]
        else:
            y = output

        # RT-DETR output should be (bs, 300, 4+nc) = (1, 300, 6)
        expected_shape = (1, 300, 6)
        test("Model output shape", y.shape == expected_shape,
             f"Expected {expected_shape}, got {y.shape}")

        return True
    except Exception as e:
        test("Model forward pass", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_model_backward(model):
    """Test backward pass of the complete model."""
    print("\n" + "="*70)
    print("TEST 6: Model Backward Pass (Gradient Flow)")
    print("="*70)

    if model is None:
        test("Model backward pass", False, "Model not loaded")
        return False

    try:
        # Test train mode forward + backward pass
        model.model.train()
        x = torch.randn(1, 3, 640, 640)

        # Forward pass
        output = model.model(x)

        test("Model train forward pass", output is not None, "Output is None")

        # Simple loss computation (just for gradient test)
        if isinstance(output, tuple):
            # In train mode, output is (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, proto_loss)
            if len(output) >= 2:
                loss = output[1].sum()  # Use dec_scores for loss
            else:
                loss = output[0].sum()
        else:
            loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients flow to key parameters
        has_grad = False
        for name, param in model.model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        test("Model gradient flow", has_grad, "No gradients found in model parameters")

        return True
    except Exception as e:
        test("Model backward pass", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("RT-DETR Plan B-Minimal Configuration Test Suite")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Run tests
    test_module_imports()
    test_repapconvblock()
    test_hcprtdetrdecoder()
    model = test_model_config()
    test_model_forward(model)
    test_model_backward(model)

    # Summary
    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*70)

    if failed > 0:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
        return 1
    else:
        print("\n🎉 All tests passed! Plan B-Minimal is ready for training.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
