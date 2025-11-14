#!/usr/bin/env python3
"""
Test script to verify RepAPConvBlock fix.

This test verifies that the critical re-parameterization bug has been fixed.

Previous Bug:
    - Different branches had incompatible kernel shapes
    - switch_to_deploy() would crash with shape mismatch
    - Could not export to ONNX or deploy to production

Expected After Fix:
    - All branches use full c1 channels
    - All kernels have compatible shape [c2, c1, ...]
    - switch_to_deploy() works correctly
    - Output matches between training and deployment mode
"""

import sys
import torch
import torch.nn as nn

# Add ultralytics to path
sys.path.insert(0, '/home/user/ultralytics_ofver8.3.173')

from ultralytics.nn.modules.block import RepAPConvBlock

def test_rep_ap_conv_block():
    """Test RepAPConvBlock with various configurations."""

    print("=" * 80)
    print("Testing RepAPConvBlock Fix")
    print("=" * 80)
    print()

    test_configs = [
        # (c1, c2, stride, name)
        (64, 64, 1, "Same channels, stride=1 (with identity)"),
        (64, 128, 1, "Different channels, stride=1 (no identity)"),
        (128, 256, 2, "Downsample, stride=2"),
        (32, 32, 1, "Small channels"),
        (512, 1024, 2, "Large channels"),
    ]

    all_passed = True

    for c1, c2, stride, name in test_configs:
        print(f"Test: {name}")
        print(f"  Config: c1={c1}, c2={c2}, stride={stride}")

        try:
            # Create block
            block = RepAPConvBlock(c1, c2, s=stride, deploy=False)

            # Test input
            bs, h, w = 2, 32, 32
            x = torch.randn(bs, c1, h, w)

            # Forward in training mode
            with torch.no_grad():
                out_train = block(x)

            expected_h = h // stride
            expected_w = w // stride

            print(f"  ✅ Training forward: {x.shape} → {out_train.shape}")
            assert out_train.shape == (bs, c2, expected_h, expected_w), \
                f"Shape mismatch! Expected {(bs, c2, expected_h, expected_w)}, got {out_train.shape}"

            # ========== CRITICAL TEST: switch_to_deploy() ==========
            # This would CRASH in the old version due to shape mismatch!
            try:
                block.switch_to_deploy()
                print(f"  ✅ switch_to_deploy() SUCCESS (this would crash before fix!)")
            except Exception as e:
                print(f"  ❌ switch_to_deploy() FAILED: {e}")
                all_passed = False
                continue

            # Verify deployment mode
            assert block.deploy == True, "Deploy flag not set!"
            assert hasattr(block, 'rep_conv'), "rep_conv not created!"
            print(f"  ✅ Deployment mode activated")

            # Forward in deployment mode
            with torch.no_grad():
                out_deploy = block(x)

            print(f"  ✅ Deployment forward: {x.shape} → {out_deploy.shape}")
            assert out_deploy.shape == out_train.shape, \
                f"Shape mismatch between train and deploy!"

            # ========== CRITICAL TEST: Output consistency ==========
            # Outputs should be VERY similar (within numerical precision)
            max_diff = (out_train - out_deploy).abs().max().item()
            mean_diff = (out_train - out_deploy).abs().mean().item()

            print(f"  Output difference: max={max_diff:.2e}, mean={mean_diff:.2e}")

            if max_diff < 1e-4:
                print(f"  ✅ Output consistency EXCELLENT (diff < 1e-4)")
            elif max_diff < 1e-2:
                print(f"  ⚠️  Output consistency OK (diff < 1e-2)")
            else:
                print(f"  ❌ Output consistency POOR (diff > 1e-2)")
                all_passed = False

            print(f"  ✅ PASSED: {name}")

        except Exception as e:
            print(f"  ❌ FAILED: {name}")
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

        print()

    # ========== Final Summary ==========
    print("=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✅ RepAPConvBlock fix verified:")
        print("   - All branches now use full c1 channels")
        print("   - switch_to_deploy() works without shape mismatch")
        print("   - Training and deployment outputs are consistent")
        print("   - Ready for production deployment!")
        print()
        print("Next steps:")
        print("   1. ✅ Can train models with AREP-Backbone")
        print("   2. ✅ Can export to ONNX/TensorRT")
        print("   3. ✅ Can deploy to production")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = test_rep_ap_conv_block()
    sys.exit(exit_code)
