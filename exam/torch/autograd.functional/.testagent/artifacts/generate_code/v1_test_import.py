#!/usr/bin/env python3
"""Simple script to test imports and basic functionality"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test G1 imports
    print("Testing G1 imports...")
    from tests.test_torch_autograd_functional_g1 import (
        simple_scalar_func,
        linear_transform_func,
        TestAutogradFunctionalG1
    )
    print("✓ G1 imports successful")
    
    # Test G2 imports
    print("\nTesting G2 imports...")
    from tests.test_torch_autograd_functional_g2 import (
        quadratic_form_func,
        TestAutogradFunctionalG2
    )
    print("✓ G2 imports successful")
    
    # Test torch.autograd.functional imports
    print("\nTesting torch.autograd.functional imports...")
    import torch
    import torch.autograd.functional as autograd_func
    print("✓ torch.autograd.functional imports successful")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    x = torch.randn(2, 2, dtype=torch.float32, requires_grad=True)
    result = simple_scalar_func(x)
    print(f"✓ simple_scalar_func works: output shape = {result.shape}, value = {result.item():.4f}")
    
    x2 = torch.randn(3, dtype=torch.float64, requires_grad=True)
    result2 = linear_transform_func(x2)
    print(f"✓ linear_transform_func works: output shape = {result2.shape}")
    
    x3 = torch.randn(2, dtype=torch.float32, requires_grad=True)
    result3 = quadratic_form_func(x3)
    print(f"✓ quadratic_form_func works: output shape = {result3.shape}, value = {result3.item():.4f}")
    
    print("\n✅ All imports and basic functionality tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)