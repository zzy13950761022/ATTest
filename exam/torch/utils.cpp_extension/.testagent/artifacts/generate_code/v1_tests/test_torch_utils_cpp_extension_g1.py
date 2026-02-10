import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# ==== BLOCK:HEADER START ====
# 导入目标函数
from torch.utils.cpp_extension import load

# 测试辅助函数和fixtures
@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run to simulate compilation"""
    with patch('subprocess.run') as mock_run:
        # 模拟成功编译
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b"Compilation successful"
        mock_result.stderr = b""
        mock_run.return_value = mock_result
        yield mock_run

@pytest.fixture
def mock_import_module():
    """Mock importlib.import_module to simulate module loading"""
    with patch('importlib.import_module') as mock_import:
        mock_module = Mock()
        mock_module.__name__ = "test_extension"
        mock_module.test_function = Mock(return_value=42)
        mock_import.return_value = mock_module
        yield mock_import

@pytest.fixture
def mock_tempfile():
    """Mock tempfile.mkdtemp to control temp directory"""
    with patch('tempfile.mkdtemp') as mock_mkdtemp:
        mock_mkdtemp.return_value = "/tmp/test_build_12345"
        yield mock_mkdtemp

@pytest.fixture
def mock_os_path():
    """Mock os.path functions"""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        yield mock_exists

@pytest.fixture
def mock_cuda_available():
    """Mock torch.cuda.is_available"""
    with patch('torch.cuda.is_available') as mock_cuda:
        mock_cuda.return_value = True
        yield mock_cuda

@pytest.fixture
def cleanup_temp_dirs():
    """Cleanup any temporary directories created during tests"""
    temp_dirs = []
    yield temp_dirs
    for dir_path in temp_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)

# 测试数据
SIMPLE_CPP_CONTENT = '''
#include <torch/extension.h>
#include <iostream>

torch::Tensor test_function(torch::Tensor input) {
    return input * 2;
}

PYBIND11_MODULE(test_extension, m) {
    m.def("test_function", &test_function, "Test function");
}
'''

CUDA_CPP_CONTENT = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
    }
}

torch::Tensor cuda_function(torch::Tensor input) {
    auto output = input.clone();
    float* data = output.data_ptr<float>();
    int size = output.numel();
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel<<<blocks, threads>>>(data, size);
    cudaDeviceSynchronize();
    
    return output;
}

PYBIND11_MODULE(cuda_extension, m) {
    m.def("cuda_function", &cuda_function, "CUDA test function");
}
'''
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
# TC-01: 基本C++扩展编译加载
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
# TC-02: 混合C++/CUDA扩展编译
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: 单字符串sources参数 (DEFERRED - 占位)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: 保留占位 (DEFERRED)
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====