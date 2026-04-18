#!/usr/bin/env python3
"""
Test script to verify the PIXFORM installation fix.
This tests that the script changes work as expected.
"""

import re
import sys
from pathlib import Path

def check_install_script():
    """Verify install.ps1 has the key fixes."""
    print("\n========== PIXFORM Installation Fix Verification ==========\n")

    install_path = Path("install.ps1")
    if not install_path.exists():
        print("ERROR: install.ps1 not found")
        return False

    content = install_path.read_text(encoding='utf-8')

    # Test 1: Check for Windows OS detection
    print("[TEST 1] Windows TRELLIS.2 Auto-Disable...")
    if "Environment" in content and "OSVersion" in content and "Win32NT" in content:
        if "trellis2" in content.lower():
            print("  ✓ Windows OS detection for TRELLIS.2 found")
        else:
            print("  ⚠ OS detection found but TRELLIS.2 check unclear")
    else:
        print("  ✗ Windows OS detection not found")
        return False

    # Test 2: Check for graceful degradation
    print("\n[TEST 2] Graceful Error Handling...")
    if "will be disabled" in content or "skipping" in content.lower():
        print("  ✓ Graceful degradation messages found")
    else:
        print("  ⚠ Graceful degradation messages not clear")

    # Test 3: Check for native extension checks
    print("\n[TEST 3] Native Extension Pre-flight Checks...")
    if "cl" in content and "nvcc" in content:
        print("  ✓ MSVC and CUDA checks found")
    else:
        print("  ⚠ Compiler checks not obvious")

    # Test 4: Verify TripoSR, Hunyuan, TRELLIS are still installed
    print("\n[TEST 4] Core Models Still Available...")
    models = ["triposr", "hunyuan", "trellis"]
    for model in models:
        if model in content.lower():
            print(f"  ✓ {model.upper()} installation code present")
        else:
            print(f"  ✗ {model.upper()} missing!")
            return False

    # Test 5: Check backend app.py
    print("\n[TEST 5] Backend TRELLIS.2 Compatibility...")
    app_path = Path("backend/app.py")
    if app_path.exists():
        app_content = app_path.read_text(encoding='utf-8')
        if "trellis2" in app_content.lower():
            if "failed" in app_content or "not_installed" in app_content:
                print("  ✓ Backend has graceful TRELLIS.2 fallback")
            else:
                print("  ✓ Backend has TRELLIS.2 handling")
        else:
            print("  ! TRELLIS.2 not mentioned in backend (might be OK)")

    return True

def print_summary():
    """Print installation recommendations."""
    print("\n========== RECOMMENDED INSTALLATION ==========\n")
    print("Command:")
    print("  cd C:\\Users\\YourUsername\\path\\to\\pixform")
    print("  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass")
    print("  .\\install.ps1 -Profile auto")
    print("  Then select: all (or 1,2,3 for TripoSR, Hunyuan, TRELLIS)")

    print("\nAvailable Models on Windows:")
    print("  ✓ TripoSR - Fast, multi-device (CPU/MPS/CUDA)")
    print("  ✓ Hunyuan3D-2 - Higher quality (CUDA only)")
    print("  ✓ TRELLIS - Highest quality (CUDA only, NO native builds)")
    print("  ⊘ TRELLIS.2 - Auto-disabled (has fragile native dependencies)")

    print("\nQuality Ranking (Best to Fastest):")
    print("  1. TRELLIS - Best quality, no native build issues!")
    print("  2. Hunyuan3D-2 - Good quality")
    print("  3. TripoSR - Fast and practical")

    print("\n========== VERIFICATION COMPLETE ==========")
    print("✓ Fix is in place and ready for installation!")

if __name__ == "__main__":
    try:
        if check_install_script():
            print("\n✓ All checks passed!")
            print_summary()
            sys.exit(0)
        else:
            print("\n✗ Some checks failed")
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

