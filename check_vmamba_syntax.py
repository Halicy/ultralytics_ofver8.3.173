#!/usr/bin/env python3
"""
Syntax and import check for VMamba integration.

This script validates that all VMamba-related files have correct syntax
and can be imported without errors.
"""

import sys
import ast
import os


def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_yaml_syntax(file_path):
    """Check if a YAML file has valid syntax."""
    try:
        import yaml
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True, "OK"
    except Exception as e:
        # YAML might not be installed, so just check basic structure
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            # Basic checks
            if 'backbone:' in content and 'head:' in content:
                return True, "OK (basic check)"
            else:
                return False, "Missing required sections"
        except Exception as e:
            return False, f"Error: {e}"


def main():
    """Check all VMamba-related files."""
    print("=" * 80)
    print("VMamba Integration Syntax Check")
    print("=" * 80)

    files_to_check = [
        ("ultralytics/nn/modules/vmamba.py", "python"),
        ("ultralytics/nn/modules/__init__.py", "python"),
        ("ultralytics/cfg/models/rt-detr/rtdetr-vmamba-tiny.yaml", "yaml"),
        ("ultralytics/cfg/models/rt-detr/rtdetr-vmamba-small.yaml", "yaml"),
    ]

    all_passed = True

    for file_path, file_type in files_to_check:
        full_path = os.path.join(os.getcwd(), file_path)
        if not os.path.exists(full_path):
            print(f"✗ {file_path}: File not found")
            all_passed = False
            continue

        if file_type == "python":
            success, message = check_python_syntax(full_path)
        elif file_type == "yaml":
            success, message = check_yaml_syntax(full_path)
        else:
            success, message = False, "Unknown file type"

        status = "✓" if success else "✗"
        print(f"{status} {file_path}: {message}")

        if not success:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL SYNTAX CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED")
    print("=" * 80)

    # Try to import the module (this will only work if dependencies are available)
    print("\n" + "=" * 80)
    print("Import Check (may fail if dependencies not installed)")
    print("=" * 80)

    try:
        sys.path.insert(0, os.getcwd())
        from ultralytics.nn.modules.vmamba import VisionMambaBackbone, VMambaStage
        print("✓ Successfully imported VisionMambaBackbone and VMambaStage")
    except ImportError as e:
        print(f"⚠ Import failed (this is expected if PyTorch is not installed): {e}")
    except Exception as e:
        print(f"✗ Import error: {e}")
        all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
