"""
共享的pytest fixtures和配置
"""
import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 共享的测试数据
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

@pytest.fixture
def simple_cpp_content():
    """返回简单的C++扩展代码"""
    return SIMPLE_CPP_CONTENT

@pytest.fixture
def cuda_cpp_content():
    """返回CUDA扩展代码"""
    return CUDA_CPP_CONTENT

@pytest.fixture
def mock_compilation_success():
    """Mock成功的编译过程"""
    with patch('subprocess.run') as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b"Compilation successful"
        mock_result.stderr = b""
        mock_run.return_value = mock_result
        yield mock_run

@pytest.fixture
def mock_module_loading():
    """Mock模块加载"""
    with patch('importlib.import_module') as mock_import:
        mock_module = Mock()
        mock_module.__name__ = "test_extension"
        mock_module.test_function = Mock(return_value=42)
        mock_import.return_value = mock_module
        yield mock_import

@pytest.fixture
def temp_source_dir(tmp_path):
    """创建临时源文件目录"""
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    return source_dir

@pytest.fixture
def mock_file_baton():
    """Mock torch.utils.file_baton.FileBaton to avoid file system operations"""
    with patch('torch.utils.file_baton.FileBaton') as mock_baton_class:
        mock_baton = Mock()
        mock_baton.try_acquire.return_value = True  # 总是成功获取锁
        mock_baton.release.return_value = None
        mock_baton_class.return_value = mock_baton
        yield mock_baton_class

@pytest.fixture
def mock_os_makedirs():
    """Mock os.makedirs for directory creation"""
    with patch('os.makedirs') as mock_makedirs:
        mock_makedirs.return_value = None
        yield mock_makedirs

@pytest.fixture
def mock_os_isdir():
    """Mock os.path.isdir"""
    with patch('os.path.isdir') as mock_isdir:
        mock_isdir.return_value = True  # 总是返回目录存在
        yield mock_isdir

@pytest.fixture(autouse=True)
def cleanup_test_env():
    """自动清理测试环境"""
    # 保存原始环境
    original_env = os.environ.copy()
    
    yield
    
    # 恢复环境
    os.environ.clear()
    os.environ.update(original_env)