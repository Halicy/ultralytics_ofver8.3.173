#!/usr/bin/env python3
"""
CRITICAL TEST: Verify all innovation modules can be imported successfully
This tests the most basic requirement - can Python even load our code?
"""

import sys
import traceback

def test_import(module_path, class_name):
    """Test if a specific class can be imported"""
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ SUCCESS: {module_path}.{class_name}")
        return True, None
    except Exception as e:
        print(f"❌ FAILED: {module_path}.{class_name}")
        print(f"   Error: {str(e)}")
        traceback.print_exc()
        return False, str(e)

def main():
    print("=" * 80)
    print("CRITICAL IMPORT TEST - Testing all 5 innovations")
    print("=" * 80)
    print()

    results = {}

    # Innovation 1: ASDA (Aspect-ratio Sensitive Deformable Attention)
    print("📦 Innovation 1: ASDA")
    print("-" * 80)
    results['ASDA'] = test_import('ultralytics.nn.modules.transformer', 'ASDA')
    results['ASDATransformerEncoder'] = test_import('ultralytics.nn.modules.transformer', 'ASDATransformerEncoder')
    results['ASDATransformerEncoderLayer'] = test_import('ultralytics.nn.modules.transformer', 'ASDATransformerEncoderLayer')
    print()

    # Innovation 2: HCP-DETR (Hierarchical Category Prototype Learning)
    print("📦 Innovation 2: HCP-DETR")
    print("-" * 80)
    results['HCPRTDETRDecoder'] = test_import('ultralytics.nn.modules.head', 'HCPRTDETRDecoder')
    print()

    # Innovation 3: DQSA (Dynamic Query Selection with Sample Awareness)
    print("📦 Innovation 3: DQSA")
    print("-" * 80)
    results['DQSARTDETRDecoder'] = test_import('ultralytics.nn.modules.head', 'DQSARTDETRDecoder')
    print()

    # Innovation 4: LWHA-KD (LightWeight Hybrid Attention with Knowledge Distillation)
    print("📦 Innovation 4: LWHA-KD")
    print("-" * 80)
    results['LightWeightHybridAttention'] = test_import('ultralytics.nn.modules.transformer', 'LightWeightHybridAttention')
    results['DistillationLoss'] = test_import('ultralytics.nn.modules.transformer', 'DistillationLoss')
    results['LWHATransformerEncoderLayer'] = test_import('ultralytics.nn.modules.transformer', 'LWHATransformerEncoderLayer')
    results['LWHATransformerEncoder'] = test_import('ultralytics.nn.modules.transformer', 'LWHATransformerEncoder')
    print()

    # Innovation 5: AREP-Backbone (Aspect-Ratio Enhanced Partial Convolution Backbone)
    print("📦 Innovation 5: AREP-Backbone")
    print("-" * 80)
    results['PartialConv'] = test_import('ultralytics.nn.modules.block', 'PartialConv')
    results['AnisotropicPConv'] = test_import('ultralytics.nn.modules.block', 'AnisotropicPConv')
    results['RepAPConvBlock'] = test_import('ultralytics.nn.modules.block', 'RepAPConvBlock')
    results['AREPStage'] = test_import('ultralytics.nn.modules.block', 'AREPStage')
    results['AREPStem'] = test_import('ultralytics.nn.modules.block', 'AREPStem')
    results['AREPDownsample'] = test_import('ultralytics.nn.modules.block', 'AREPDownsample')
    print()

    # Summary
    print("=" * 80)
    print("IMPORT TEST SUMMARY")
    print("=" * 80)

    success_count = sum(1 for success, _ in results.values() if success)
    total_count = len(results)

    print(f"Total modules tested: {total_count}")
    print(f"Successful imports: {success_count}")
    print(f"Failed imports: {total_count - success_count}")
    print()

    if success_count == total_count:
        print("🎉 ALL IMPORTS SUCCESSFUL!")
        return 0
    else:
        print("⚠️  SOME IMPORTS FAILED - CODE HAS CRITICAL ERRORS")
        print("\nFailed modules:")
        for name, (success, error) in results.items():
            if not success:
                print(f"  - {name}: {error}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
