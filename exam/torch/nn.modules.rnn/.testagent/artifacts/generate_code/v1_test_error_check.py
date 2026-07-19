import torch
import torch.nn as nn

# Test 1: 2D input when 3D is expected
print("Test 1: 2D input when 3D is expected")
try:
    rnn = nn.RNN(input_size=10, hidden_size=20)
    x_2d = torch.randn(5, 10)  # Should be (seq_len, batch, input_size) or (batch, seq_len, input_size)
    output, h_n = rnn(x_2d)
    print("No error raised - unexpected!")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")

print("\n" + "="*50 + "\n")

# Test 2: Wrong input size
print("Test 2: Wrong input size")
try:
    rnn = nn.RNN(input_size=10, hidden_size=20)
    x_wrong_size = torch.randn(5, 3, 15)  # input_size=15, expected 10
    output, h_n = rnn(x_wrong_size)
    print("No error raised - unexpected!")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")

print("\n" + "="*50 + "\n")

# Test 3: Check if PyTorch supports unbatched 2D input
print("Test 3: Check unbatched 2D input (L, H_in)")
try:
    rnn = nn.RNN(input_size=10, hidden_size=20)
    x_unbatched = torch.randn(5, 10)  # (L, H_in) - unbatched input
    output, h_n = rnn(x_unbatched)
    print("Success! PyTorch supports unbatched 2D input")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {h_n.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")