#!/usr/bin/env python3
"""Quick test to verify the fixes."""

import torch
import torch.nn as nn
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

print("Testing cutoffs storage...")
# Test that cutoffs includes n_classes
model = AdaptiveLogSoftmaxWithLoss(
    in_features=5,
    n_classes=30,
    cutoffs=[5, 15],
    div_value=4.0,
    head_bias=False
)

print(f"model.cutoffs: {model.cutoffs}")
print(f"Expected: [5, 15, 30]")
assert model.cutoffs == [5, 15, 30], f"Expected [5, 15, 30], got {model.cutoffs}"
print("✓ cutoffs storage test passed")

print("\nTesting in_features=0...")
# Test if in_features=0 raises an exception
try:
    model2 = AdaptiveLogSoftmaxWithLoss(
        in_features=0,
        n_classes=10,
        cutoffs=[5],
        div_value=4.0,
        head_bias=False
    )
    print(f"in_features=0 did not raise exception (model created)")
    print(f"model2.in_features: {model2.in_features}")
except Exception as e:
    print(f"in_features=0 raised exception: {type(e).__name__}: {e}")

print("\nTesting log_prob with non-batched input...")
model3 = AdaptiveLogSoftmaxWithLoss(
    in_features=5,
    n_classes=10,
    cutoffs=[5],
    div_value=2.0,
    head_bias=False
)

# Test batched input (should work)
input_batch = torch.randn(3, 5)
try:
    log_probs_batch = model3.log_prob(input_batch)
    print(f"log_prob with batched input worked: shape {log_probs_batch.shape}")
except Exception as e:
    print(f"log_prob with batched input failed: {type(e).__name__}: {e}")

# Test non-batched input (may fail)
input_single = torch.randn(5)
try:
    log_probs_single = model3.log_prob(input_single)
    print(f"log_prob with non-batched input worked: shape {log_probs_single.shape}")
except Exception as e:
    print(f"log_prob with non-batched input failed (expected): {type(e).__name__}: {e}")

print("\nAll tests completed!")