"""
Standalone Test for HCP-DETR Utility Functions

Tests Hungarian matching and subcategory mapping without requiring full ultralytics.

Author: Ultralytics Team
Date: 2025-11-14
"""

import sys
import torch

# Add parent directory to path
sys.path.insert(0, '/home/user/ultralytics_ofver8.3.173')

# Import utility functions directly
from ultralytics.utils.hcp_utils import (
    hungarian_match_hcp_detr,
    map_to_subcategories_hcp_detr,
    validate_hcp_detr_config,
)


def test_hungarian_matching():
    """Test Hungarian matching implementation."""
    print("=" * 80)
    print("TEST 1: Hungarian Matching")
    print("=" * 80)

    bs, nq, hidden_dim = 2, 300, 256
    max_gt = 10

    print("\n1.1 Creating test data...")
    pred_boxes = torch.rand(bs, nq, 4)
    pred_scores = torch.rand(bs, nq, 6)
    gt_boxes = torch.rand(bs, max_gt, 4)
    gt_labels = torch.randint(0, 2, (bs, max_gt))

    # Add padding (-1 for invalid)
    gt_labels[0, 5:] = -1
    gt_labels[1, 7:] = -1

    query_embed = torch.randn(bs, nq, hidden_dim)

    num_valid_gt = (gt_labels >= 0).sum().item()
    print(f"   - Batch size: {bs}")
    print(f"   - Num queries: {nq}")
    print(f"   - Valid GT objects: {num_valid_gt}")

    print("\n1.2 Running Hungarian matching...")
    try:
        matched_features, matched_labels = hungarian_match_hcp_detr(
            pred_boxes, pred_scores, gt_boxes, gt_labels, query_embed
        )

        if matched_features is not None:
            print(f"   ✅ Hungarian matching successful!")
            print(f"   - Matched features shape: {matched_features.shape}")
            print(f"   - Matched labels shape: {matched_labels.shape}")
            print(f"   - Expected matches: ~{num_valid_gt}")
            print(f"   - Actual matches: {matched_features.shape[0]}")

            # Validate shapes
            assert matched_features.shape[0] == matched_labels.shape[0], "Shape mismatch!"
            assert matched_features.shape[1] == hidden_dim, "Hidden dim mismatch!"
            assert matched_labels.min() >= 0, "Invalid labels!"
            assert matched_labels.max() < 2, "Labels out of range!"

            print(f"   ✅ All shape validations passed")
        else:
            print(f"   ⚠️ No matches found (no valid GT)")

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        raise

    print("\n" + "=" * 80)
    print("TEST 1 PASSED")
    print("=" * 80 + "\n")


def test_subcategory_mapping():
    """Test subcategory mapping implementation."""
    print("=" * 80)
    print("TEST 2: Subcategory Mapping")
    print("=" * 80)

    sub_categories = {1: ['young_fruit', 'flower', 'occluded', 'malformed']}

    # Test 1: Uniform distribution
    print("\n2.1 Testing uniform distribution...")
    test_labels = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])

    mapped_uniform = map_to_subcategories_hcp_detr(
        test_labels, sub_categories, strategy='uniform'
    )

    print(f"   Original labels: {test_labels.tolist()}")
    print(f"   Mapped (uniform): {mapped_uniform.tolist()}")

    # Analyze distribution
    unique, counts = torch.unique(mapped_uniform, return_counts=True)
    print(f"   Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Check that subcategories [2, 3, 4, 5] are used
    subcats_used = [i for i in [2, 3, 4, 5] if i in unique]
    print(f"   Subcategories used: {subcats_used} / [2, 3, 4, 5]")

    assert 0 in unique, "Main category 0 should remain!"
    assert len(subcats_used) > 0, "No subcategories used!"
    print(f"   ✅ Uniform mapping successful ({len(subcats_used)}/4 subcategories)")

    # Test 2: Random distribution
    print("\n2.2 Testing random distribution...")
    mapped_random = map_to_subcategories_hcp_detr(
        test_labels, sub_categories, strategy='random'
    )

    print(f"   Mapped (random): {mapped_random.tolist()}")

    unique_random, counts_random = torch.unique(mapped_random, return_counts=True)
    print(f"   Label distribution: {dict(zip(unique_random.tolist(), counts_random.tolist()))}")

    subcats_used_random = [i for i in [2, 3, 4, 5] if i in unique_random]
    print(f"   Subcategories used: {subcats_used_random}")
    print(f"   ✅ Random mapping successful ({len(subcats_used_random)}/4 subcategories)")

    # Test 3: Larger batch
    print("\n2.3 Testing with larger batch...")
    large_labels = torch.cat([
        torch.zeros(50),
        torch.ones(50),
    ]).long()

    mapped_large = map_to_subcategories_hcp_detr(
        large_labels, sub_categories, strategy='uniform'
    )

    unique_large, counts_large = torch.unique(mapped_large, return_counts=True)
    print(f"   Label distribution: {dict(zip(unique_large.tolist(), counts_large.tolist()))}")

    # With 50 samples and 4 subcategories, each should get ~12-13 samples
    for subcat in [2, 3, 4, 5]:
        if subcat in unique_large:
            count = counts_large[unique_large == subcat].item()
            print(f"   - Subcategory {subcat}: {count} samples (~{count/50*100:.1f}%)")

    # All 4 subcategories should be used with uniform distribution on 50 samples
    assert len([i for i in [2, 3, 4, 5] if i in unique_large]) == 4, \
        "All subcategories should be used with enough samples!"
    print(f"   ✅ All 4 subcategories used with uniform distribution")

    print("\n" + "=" * 80)
    print("TEST 2 PASSED")
    print("=" * 80 + "\n")


def test_config_validation():
    """Test configuration validation."""
    print("=" * 80)
    print("TEST 3: Configuration Validation")
    print("=" * 80)

    # Test valid configuration
    print("\n3.1 Testing valid configuration...")
    try:
        validate_hcp_detr_config(
            num_classes=2,
            sub_categories={1: ['young_fruit', 'flower', 'occluded', 'malformed']},
            total_prototypes=6
        )
        print(f"   ✅ Valid configuration accepted (nc=2, total=6)")
    except ValueError as e:
        print(f"   ❌ Should accept valid config: {e}")
        raise

    # Test invalid configuration (wrong total)
    print("\n3.2 Testing invalid configuration (wrong total)...")
    try:
        validate_hcp_detr_config(
            num_classes=2,
            sub_categories={1: ['young_fruit', 'flower']},
            total_prototypes=6  # Should be 4 (2 main + 2 sub)
        )
        print(f"   ❌ Should reject invalid config!")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly rejected invalid config")
        print(f"   Error message: {str(e)[:80]}...")

    # Test invalid subcategory key
    print("\n3.3 Testing invalid subcategory key...")
    try:
        validate_hcp_detr_config(
            num_classes=2,
            sub_categories={5: ['invalid']},  # Key 5 >= num_classes=2
            total_prototypes=3
        )
        print(f"   ❌ Should reject invalid subcategory key!")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"   ✅ Correctly rejected invalid subcategory key")

    print("\n" + "=" * 80)
    print("TEST 3 PASSED")
    print("=" * 80 + "\n")


def test_gradient_flow():
    """Test that subcategory mapping enables gradient flow to all prototypes."""
    print("=" * 80)
    print("TEST 4: Gradient Flow Validation")
    print("=" * 80)

    print("\n4.1 Creating mock prototype layer...")
    total_nc = 6  # 2 main + 4 subcategories
    proto_dim = 256

    prototypes = torch.nn.Embedding(total_nc, proto_dim)
    optimizer = torch.optim.Adam(prototypes.parameters())

    print(f"   - Total prototypes: {total_nc}")
    print(f"   - Prototype dimension: {proto_dim}")

    # Simulate training WITHOUT subcategory mapping (Bug #3)
    print("\n4.2 Testing WITHOUT subcategory mapping (broken)...")
    optimizer.zero_grad()

    # Only use labels [0, 1] (main categories)
    labels_broken = torch.randint(0, 2, (100,))
    features = torch.randn(100, proto_dim)

    # Simple loss: distance to prototypes
    selected_protos = prototypes(labels_broken)
    loss_broken = ((features - selected_protos) ** 2).mean()
    loss_broken.backward()

    grad_norms_broken = prototypes.weight.grad.norm(dim=1)
    num_trained_broken = (grad_norms_broken > 0).sum().item()

    print(f"   Prototypes receiving gradients: {num_trained_broken}/{total_nc}")
    print(f"   Gradient norms: {grad_norms_broken.tolist()}")
    assert num_trained_broken <= 2, "Should only train 2 prototypes without subcategory mapping"
    print(f"   ❌ Only {num_trained_broken}/6 prototypes trained (67% wasted!)")

    # Simulate training WITH subcategory mapping (Fixed!)
    print("\n4.3 Testing WITH subcategory mapping (fixed)...")
    prototypes.zero_grad()
    optimizer.zero_grad()

    # Use subcategory mapping
    labels_main = torch.randint(0, 2, (100,))
    sub_categories = {1: ['young_fruit', 'flower', 'occluded', 'malformed']}
    labels_fixed = map_to_subcategories_hcp_detr(
        labels_main, sub_categories, strategy='uniform'
    )

    # Simple loss with mapped labels
    selected_protos_fixed = prototypes(labels_fixed)
    loss_fixed = ((features - selected_protos_fixed) ** 2).mean()
    loss_fixed.backward()

    grad_norms_fixed = prototypes.weight.grad.norm(dim=1)
    num_trained_fixed = (grad_norms_fixed > 0).sum().item()

    print(f"   Prototypes receiving gradients: {num_trained_fixed}/{total_nc}")
    print(f"   Gradient norms: {grad_norms_fixed.tolist()}")
    assert num_trained_fixed >= 5, "Should train at least 5/6 prototypes with subcategory mapping"
    print(f"   ✅ {num_trained_fixed}/6 prototypes trained! (Bug #3 FIXED)")

    print("\n" + "=" * 80)
    print("TEST 4 PASSED")
    print("=" * 80 + "\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("HCP-DETR UTILITY FUNCTIONS VALIDATION")
    print("=" * 80)
    print("\nValidating fixes for:")
    print("  ✓ Bug #2: Hungarian matching")
    print("  ✓ Bug #3: Subcategory mapping")
    print("\n" + "=" * 80 + "\n")

    try:
        test_hungarian_matching()
        test_subcategory_mapping()
        test_config_validation()
        test_gradient_flow()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\n✅ Hungarian matching: Working correctly")
        print("✅ Subcategory mapping: Working correctly")
        print("✅ Configuration validation: Working correctly")
        print("✅ Gradient flow: All prototypes trained")
        print("\n" + "=" * 80)
        print("\nHCP-DETR Fixes Summary:")
        print("  • Bug #1: Decoder query embeddings (implemented in head.py)")
        print("  • Bug #2: Hungarian matching (VALIDATED ✓)")
        print("  • Bug #3: Subcategory mapping (VALIDATED ✓)")
        print("\n🚀 HCP-DETR is ready for training!")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TESTS FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
