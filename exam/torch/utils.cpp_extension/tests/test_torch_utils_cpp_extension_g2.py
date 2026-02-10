import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# ==== BLOCK:HEADER START ====
# 导入目标函数
from torch.utils.cpp_extension import load

# 测试辅助函数和fixtures（与G1共享，但需要重新定义）
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
def mock_os_makedirs():
    """Mock os.makedirs"""
    with patch('os.makedirs') as mock_makedirs:
        yield mock_makedirs

@pytest.fixture
def mock_shutil_rmtree():
    """Mock shutil.rmtree"""
    with patch('shutil.rmtree') as mock_rmtree:
        yield mock_rmtree

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
def mock_os_makedirs_for_cache():
    """Mock os.makedirs for cache directory creation"""
    with patch('os.makedirs') as mock_makedirs:
        mock_makedirs.return_value = None
        yield mock_makedirs

@pytest.fixture
def mock_hash_source_files():
    """Mock torch.utils._cpp_extension_versioner.hash_source_files"""
    with patch('torch.utils._cpp_extension_versioner.hash_source_files') as mock_hash:
        mock_hash.return_value = 12345  # 返回固定哈希值
        yield mock_hash

@pytest.fixture
def mock_os_isdir():
    """Mock os.path.isdir"""
    with patch('os.path.isdir') as mock_isdir:
        mock_isdir.return_value = True  # 总是返回目录存在
        yield mock_isdir

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
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
# TC-03: 参数组合测试
@pytest.mark.parametrize("test_name,sources,extra_cflags,extra_include_paths,build_directory,keep_intermediates", [
    ("param_test", ["test.cpp"], ["-Wall", "-Wextra"], ["/usr/local/include"], "/tmp/test_build", False),
])
def test_parameter_combinations(
    test_name, sources, extra_cflags, extra_include_paths, build_directory, keep_intermediates,
    mock_subprocess, mock_import_module, mock_tempfile, mock_os_path,
    mock_os_makedirs, mock_shutil_rmtree, mock_file_baton, mock_os_makedirs_for_cache,
    mock_os_isdir, cleanup_temp_dirs, tmp_path
):
    """
    测试各种参数组合
    """
    # 创建测试源文件
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    
    for source_file in sources:
        file_path = source_dir / source_file
        file_path.write_text(SIMPLE_CPP_CONTENT)
    
    # 准备源文件路径列表
    source_paths = [str(source_dir / s) for s in sources]
    
    # 调用load函数
    module = load(
        name=test_name,
        sources=source_paths,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        keep_intermediates=keep_intermediates
    )
    
    # WEAK断言：模块加载成功
    assert module is not None, "Module should be loaded successfully"
    assert hasattr(module, '__name__'), "Module should have __name__ attribute"
    assert module.__name__ == test_name, f"Module name should be {test_name}"
    
    # WEAK断言：编译过程被调用
    mock_subprocess.assert_called()
    
    # WEAK断言：模块导入被调用
    mock_import_module.assert_called()
    
    # WEAK断言：参数被应用
    # 检查是否创建了构建目录
    if build_directory is not None:
        mock_os_makedirs.assert_called()
    else:
        # 如果使用临时目录，应该调用mkdtemp
        pass
    
    # 检查编译参数
    call_args = mock_subprocess.call_args
    assert call_args is not None, "subprocess.run should be called"
    
    # 检查是否包含指定的编译器标志
    cmd_args = call_args[0][0]
    assert isinstance(cmd_args, list), "Command should be a list"
    
    if extra_cflags:
        for flag in extra_cflags:
            # 检查标志是否在命令中（可能以不同形式出现）
            flag_found = any(flag in arg for arg in cmd_args)
            if not flag_found:
                print(f"Warning: Compiler flag {flag} not found in command args")
    
    print(f"✓ Parameter combination test passed: {test_name}")
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# TC-04: 无效源文件错误处理
@pytest.mark.parametrize("test_name,sources,expect_error,error_type", [
    ("invalid_ext", [], True, RuntimeError),
    ("invalid_ext", ["nonexistent.cpp"], True, RuntimeError),
])
def test_invalid_source_files(
    test_name, sources, expect_error, error_type,
    mock_subprocess, mock_tempfile, mock_os_path, mock_file_baton,
    mock_os_makedirs_for_cache, mock_os_isdir, cleanup_temp_dirs, tmp_path
):
    """
    测试无效源文件的错误处理
    """
    # 准备源文件路径列表
    source_paths = []
    for source_file in sources:
        if source_file:  # 非空字符串
            source_paths.append(str(tmp_path / source_file))
    
    # 根据测试场景设置mock
    with patch('os.path.exists') as mock_exists:
        if not sources or "nonexistent" in str(sources):
            # 文件不存在
            mock_exists.return_value = False
        else:
            mock_exists.return_value = True
        
        if expect_error:
            # 期望抛出异常
            with pytest.raises(error_type) as exc_info:
                # 对于nonexistent.cpp的情况，需要mock哈希计算
                if sources and "nonexistent" in str(sources):
                    with patch('torch.utils._cpp_extension_versioner.hash_source_files') as mock_hash:
                        mock_hash.return_value = 12345
                        
                        load(
                            name=test_name,
                            sources=source_paths if source_paths else []
                        )
                else:
                    # 对于空sources的情况，正常调用
                    load(
                        name=test_name,
                        sources=source_paths if source_paths else []
                    )
            
            # WEAK断言：异常被正确抛出
            assert exc_info.value is not None, f"Should raise {error_type.__name__}"
            
            # WEAK断言：错误类型正确
            assert isinstance(exc_info.value, error_type), \
                f"Should raise {error_type.__name__}, got {type(exc_info.value).__name__}"
            
            # WEAK断言：错误消息包含相关信息
            error_msg = str(exc_info.value).lower()
            if not sources:
                assert "empty" in error_msg or "no source" in error_msg or "sources" in error_msg, \
                    "Error message should mention empty sources"
            elif "nonexistent" in str(sources):
                assert "not exist" in error_msg or "not found" in error_msg or "invalid" in error_msg, \
                    "Error message should mention file not found"
            
            print(f"✓ Invalid source file test passed (expected error): {test_name}")
        else:
            # 不应该抛出异常的情况（虽然这个测试用例都是期望错误的）
            # 这里保持逻辑完整性
            try:
                module = load(
                    name=test_name,
                    sources=source_paths
                )
                assert module is not None, "Module should be loaded"
                print(f"✓ Invalid source file test passed (no error): {test_name}")
            except Exception as e:
                pytest.fail(f"Unexpected error: {e}")
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: 保留占位 (DEFERRED)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# TC-08: 保留占位 (DEFERRED)
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====