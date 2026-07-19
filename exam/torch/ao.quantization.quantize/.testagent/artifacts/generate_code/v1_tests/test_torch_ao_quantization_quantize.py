"""
Main test file for torch.ao.quantization.quantize
This file imports tests from group files for backward compatibility.
"""
import pytest

# Import tests from group files
pytest.main(["-v", "tests/test_torch_ao_quantization_quantize_g1.py"])
pytest.main(["-v", "tests/test_torch_ao_quantization_quantize_g2.py"])

if __name__ == "__main__":
    print("Run tests using: pytest tests/test_torch_ao_quantization_quantize_g1.py")
    print("Run tests using: pytest tests/test_torch_ao_quantization_quantize_g2.py")