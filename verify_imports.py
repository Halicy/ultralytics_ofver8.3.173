#!/usr/bin/env python3
"""
Comprehensive Import and Module Verification Script
====================================================
This script verifies all imports and module integrations for Plan B-Minimal.
"""

import sys
import traceback

def test_section(name):
    print(f"\n{'='*60}")
    print(f" {name}")
    print('='*60)

def test_pass(msg):
    print(f"✅ PASS: {msg}")

def test_fail(msg, error=""):
    print(f"❌ FAIL: {msg}")
    if error:
        print(f"   Error: {error}")

# Test 1: Basic PyTorch Import
test_section("1. Basic PyTorch Import")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    test_pass(f"PyTorch {torch.__version__} imported successfully")
except Exception as e:
    test_fail("PyTorch import", str(e))
    sys.exit(1)

# Test 2: Ultralytics Core Imports
test_section("2. Ultralytics Core Imports")
try:
    from ultralytics.nn.modules import Conv, RepC3, RTDETRDecoder
    test_pass("Core modules imported (Conv, RepC3, RTDETRDecoder)")
except Exception as e:
    test_fail("Core module import", str(e))
    traceback.print_exc()

# Test 3: New Innovation Module Imports
test_section("3. Innovation Module Imports")
try:
    from ultralytics.nn.modules import RepAPConvBlock
    test_pass("RepAPConvBlock imported successfully")
except Exception as e:
    test_fail("RepAPConvBlock import", str(e))
    traceback.print_exc()

try:
    from ultralytics.nn.modules import HCPRTDETRDecoder
    test_pass("HCPRTDETRDecoder imported successfully")
except Exception as e:
    test_fail("HCPRTDETRDecoder import", str(e))
    traceback.print_exc()

# Test 4: Loss Function Imports
test_section("4. Loss Function Imports")
try:
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    test_pass("RTDETRDetectionLoss imported successfully")
except Exception as e:
    test_fail("RTDETRDetectionLoss import", str(e))
    traceback.print_exc()

try:
    from ultralytics.models.utils.loss import HCPRTDETRDetectionLoss
    test_pass("HCPRTDETRDetectionLoss imported successfully")
except Exception as e:
    test_fail("HCPRTDETRDetectionLoss import", str(e))
    traceback.print_exc()

# Test 5: Model Class Imports
test_section("5. Model Class Imports")
try:
    from ultralytics.nn.tasks import RTDETRDetectionModel
    test_pass("RTDETRDetectionModel imported successfully")
except Exception as e:
    test_fail("RTDETRDetectionModel import", str(e))
    traceback.print_exc()

try:
    from ultralytics.nn.tasks import HCPRTDETRDetectionModel
    test_pass("HCPRTDETRDetectionModel imported successfully")
except Exception as e:
    test_fail("HCPRTDETRDetectionModel import", str(e))
    traceback.print_exc()

# Test 6: RepAPConvBlock Instantiation
test_section("6. RepAPConvBlock Instantiation and Forward")
try:
    from ultralytics.nn.modules import RepAPConvBlock
    block = RepAPConvBlock(c1=256, c2=256, n=3, e=0.5, shortcut=True)
    test_pass("RepAPConvBlock instantiation successful")

    # Test forward pass
    x = torch.randn(2, 256, 40, 40)
    y = block(x)
    assert y.shape == (2, 256, 40, 40), f"Shape mismatch: {y.shape}"
    test_pass(f"Forward pass: {x.shape} -> {y.shape}")

    # Test gradient flow
    loss = y.sum()
    loss.backward()
    assert block.alpha.grad is not None, "No gradient for alpha"
    test_pass(f"Gradient flow OK, alpha weight: {block.get_aspect_ratio_weight():.4f}")

except Exception as e:
    test_fail("RepAPConvBlock instantiation/forward", str(e))
    traceback.print_exc()

# Test 7: HCPRTDETRDecoder Instantiation
test_section("7. HCPRTDETRDecoder Instantiation")
try:
    from ultralytics.nn.modules import HCPRTDETRDecoder
    decoder = HCPRTDETRDecoder(
        nc=2, ch=(256, 256, 256), hd=256, nq=300, ndp=4, nh=8, ndl=6,
        d_ffn=1024, dropout=0.0, act=nn.ReLU(), eval_idx=-1, nd=100,
        label_noise_ratio=0.5, box_noise_scale=1.0, learnt_init_query=False,
        num_prototypes=6, prototype_dim=128, prototype_temp=0.07, prototype_loss_weight=0.3
    )
    test_pass("HCPRTDETRDecoder instantiation successful")

    # Check attributes
    assert hasattr(decoder, 'prototypes'), "Missing prototypes"
    assert decoder.prototypes.shape == (6, 256), f"Wrong shape: {decoder.prototypes.shape}"
    test_pass(f"Prototypes shape: {decoder.prototypes.shape}")

    assert hasattr(decoder, 'sub_to_main_map'), "Missing sub_to_main_map"
    test_pass(f"Sub-to-main mapping: {decoder.sub_to_main_map.shape}")

except Exception as e:
    test_fail("HCPRTDETRDecoder instantiation", str(e))
    traceback.print_exc()

# Test 8: HCPRTDETRDetectionLoss Instantiation
test_section("8. HCPRTDETRDetectionLoss Instantiation")
try:
    from ultralytics.models.utils.loss import HCPRTDETRDetectionLoss
    criterion = HCPRTDETRDetectionLoss(nc=2, use_vfl=True, asrw_warmup=10, asrw_weight=0.5)
    test_pass("HCPRTDETRDetectionLoss instantiation successful")

    # Test ASRW methods
    criterion.set_epoch(50)
    criterion.set_max_epochs(150)
    test_pass(f"ASRW epoch set: {criterion.current_epoch}/{criterion.max_epochs}")

except Exception as e:
    test_fail("HCPRTDETRDetectionLoss instantiation", str(e))
    traceback.print_exc()

# Test 9: YAML Configuration Loading
test_section("9. YAML Configuration Loading")
try:
    import yaml
    config_path = "ultralytics/cfg/models/rt-detr/rtdetr-l-plan-b-minimal.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    test_pass(f"YAML loaded: {config_path}")

    # Check structure
    assert 'nc' in config, "Missing nc"
    assert 'backbone' in config, "Missing backbone"
    assert 'head' in config, "Missing head"
    test_pass(f"Config structure: nc={config['nc']}, backbone={len(config['backbone'])} layers, head={len(config['head'])} layers")

    # Check for RepAPConvBlock
    repap_count = 0
    hcp_count = 0
    for layer in config['head']:
        if len(layer) >= 3:
            if layer[2] == 'RepAPConvBlock':
                repap_count += 1
            if layer[2] == 'HCPRTDETRDecoder':
                hcp_count += 1
    test_pass(f"Found {repap_count} RepAPConvBlock layers and {hcp_count} HCPRTDETRDecoder")

except Exception as e:
    test_fail("YAML config loading", str(e))
    traceback.print_exc()

# Test 10: Full Model Loading (Critical Test)
test_section("10. Full Model Loading from YAML")
try:
    from ultralytics import RTDETR
    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-l-plan-b-minimal.yaml")
    test_pass("RTDETR model loaded from Plan B-Minimal config")

    # Check model structure
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    test_pass(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Check for innovation modules
    has_repap = False
    has_hcp = False
    for name, module in model.model.named_modules():
        if 'RepAPConvBlock' in str(type(module)):
            has_repap = True
        if 'HCPRTDETRDecoder' in str(type(module)):
            has_hcp = True

    if has_repap:
        test_pass("RepAPConvBlock found in model")
    else:
        test_fail("RepAPConvBlock NOT found in model")

    if has_hcp:
        test_pass("HCPRTDETRDecoder found in model")
    else:
        test_fail("HCPRTDETRDecoder NOT found in model")

except Exception as e:
    test_fail("Full model loading", str(e))
    traceback.print_exc()

# Test 11: Forward Pass Test
test_section("11. Model Forward Pass (Inference Mode)")
try:
    from ultralytics import RTDETR
    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-l-plan-b-minimal.yaml")
    model.model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model.model(x)

    if isinstance(output, tuple):
        y = output[0]
    else:
        y = output

    expected_shape = (1, 300, 6)  # (bs, num_queries, 4+nc)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    test_pass(f"Inference forward pass: input {x.shape} -> output {y.shape}")

except Exception as e:
    test_fail("Model forward pass", str(e))
    traceback.print_exc()

print("\n" + "="*60)
print(" VERIFICATION COMPLETE")
print("="*60)
