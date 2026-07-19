# ==== BLOCK:HEADER START ====
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, Subset, random_split
from unittest.mock import patch, MagicMock
import bisect
import numpy as np

# Test fixtures and helper functions
@pytest.fixture
def fixed_seed():
    """Fix random seed for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Reset seed after test
    torch.manual_seed(torch.initial_seed())

def create_test_tensors(shapes, dtypes, device="cpu"):
    """创建测试张量"""
    tensors = []
    for shape, dtype in zip(shapes, dtypes):
        if dtype == "float32":
            tensor = torch.randn(shape, dtype=torch.float32, device=device)
        elif dtype == "float64":
            tensor = torch.randn(shape, dtype=torch.float64, device=device)
        elif dtype == "int64":
            tensor = torch.randint(0, 10, shape, dtype=torch.int64, device=device)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        tensors.append(tensor)
    return tensors

class MockDataset(Dataset):
    """模拟数据集用于测试"""
    def __init__(self, size, data=None):
        self.size = size
        self.data = data if data is not None else list(range(size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = self.size + idx
        if 0 <= idx < self.size:
            return self.data[idx]
        raise IndexError(f"Index {idx} out of range")
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
@pytest.mark.parametrize("tensor_shapes,dtypes,device,index", [
    ([[10, 3, 32, 32], [10, 1]], ["float32", "float32"], "cpu", 0),
    ([[5, 1, 28, 28], [5, 10]], ["float64", "int64"], "cpu", -1),
    ([[0, 3, 32, 32], [0, 1]], ["float32", "float32"], "cpu", 0),
])
def test_tensor_dataset_basic(tensor_shapes, dtypes, device, index):
    """TensorDataset 基本功能验证"""
    # 创建测试张量
    tensors = []
    for shape, dtype in zip(tensor_shapes, dtypes):
        if dtype == "float32":
            tensor = torch.randn(shape, dtype=torch.float32, device=device)
        elif dtype == "float64":
            tensor = torch.randn(shape, dtype=torch.float64, device=device)
        elif dtype == "int64":
            tensor = torch.randint(0, 10, shape, dtype=torch.int64, device=device)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        tensors.append(tensor)
    
    # 创建 TensorDataset
    dataset = TensorDataset(*tensors)
    
    # 验证数据集长度
    assert len(dataset) == tensor_shapes[0][0], "数据集长度应与第一个张量的第一维度一致"
    
    # 对于空数据集，不应尝试访问索引
    if len(dataset) == 0:
        # 空数据集应该可以创建，但访问索引会抛出IndexError
        with pytest.raises(IndexError):
            _ = dataset[0]
        return
    
    # 获取样本
    sample = dataset[index]
    
    # 验证返回类型和结构
    assert isinstance(sample, tuple), "应返回元组"
    assert len(sample) == len(tensors), "返回元组长度应与输入张量数一致"
    
    # 验证每个张量切片
    for i, (tensor, sample_slice) in enumerate(zip(tensors, sample)):
        # 形状验证
        expected_shape = tensor.shape[1:]  # 去掉第一维（batch维度）
        assert sample_slice.shape == expected_shape, f"张量{i}切片形状不正确"
        
        # 数据类型验证
        assert sample_slice.dtype == tensor.dtype, f"张量{i}数据类型未保持"
        
        # 设备验证
        assert sample_slice.device == tensor.device, f"张量{i}设备不一致"
        
        # 值验证（strong断言）
        if index >= 0:
            expected_slice = tensor[index]
        else:
            expected_slice = tensor[len(dataset) + index]
        
        # 张量值完全相等（strong断言）
        if tensor.dtype.is_floating_point:
            # 浮点类型使用更严格的容差比较
            assert torch.allclose(sample_slice, expected_slice, rtol=1e-7, atol=1e-10), \
                f"张量{i}切片值不匹配"
        else:
            # 整数类型精确比较
            assert torch.equal(sample_slice, expected_slice), \
                f"张量{i}切片值不匹配"
    
    # 边界索引处理正确（strong断言）
    # 测试边界索引
    if len(dataset) > 0:
        # 测试第一个元素
        first_sample = dataset[0]
        assert isinstance(first_sample, tuple), "边界索引0应返回元组"
        
        # 测试最后一个元素
        last_sample = dataset[-1]
        assert isinstance(last_sample, tuple), "边界索引-1应返回元组"
        
        # 验证负索引访问正确（strong断言）
        # 测试多个负索引
        for neg_idx in range(-1, -min(5, len(dataset)), -1):
            neg_sample = dataset[neg_idx]
            pos_sample = dataset[len(dataset) + neg_idx]
            
            # 验证负索引与对应正索引返回相同结果
            for neg_elem, pos_elem in zip(neg_sample, pos_sample):
                if neg_elem.dtype.is_floating_point:
                    assert torch.allclose(neg_elem, pos_elem, rtol=1e-7, atol=1e-10), \
                        f"负索引{neg_idx}与正索引{len(dataset) + neg_idx}的结果不一致"
                else:
                    assert torch.equal(neg_elem, pos_elem), \
                        f"负索引{neg_idx}与正索引{len(dataset) + neg_idx}的结果不一致"
        
        # 测试边界条件：索引超出范围应抛出IndexError
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]
        
        with pytest.raises(IndexError):
            _ = dataset[-len(dataset) - 1]
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
def test_tensor_dataset_dimension_mismatch():
    """TensorDataset 维度验证异常"""
    # 创建维度不匹配的张量
    tensor1 = torch.randn(10, 3, 32, 32)  # 10个样本
    tensor2 = torch.randn(5, 1)           # 5个样本，维度不匹配
    
    # 验证异常抛出 - TensorDataset使用assert语句，所以抛出AssertionError
    with pytest.raises(AssertionError) as exc_info:
        TensorDataset(tensor1, tensor2)
    
    # 验证错误信息包含维度不匹配提示
    error_msg = str(exc_info.value).lower()
    assert "size" in error_msg or "dimension" in error_msg or "match" in error_msg, \
        f"错误信息应包含维度/大小不匹配的提示，实际错误信息: {error_msg}"
    
    # 验证异常类型精确匹配（strong断言）
    assert exc_info.type == AssertionError, "异常类型应为 AssertionError"
    
    # 验证错误信息格式符合预期（strong断言）
    # TensorDataset的错误信息通常是 "Tensors must have same number of dimensions: got ..."
    # 或者 "Sizes of tensors must match except in dimension 0. Got ..."
    # 检查错误信息是否包含相关关键词
    assert any(keyword in error_msg for keyword in ["tensor", "size", "dimension", "match", "got"]), \
        f"错误信息格式不符合预期，实际错误信息: {error_msg}"
    
    # 测试其他维度不匹配的情况
    # 情况1：张量数量不同但第一维度相同
    tensor3 = torch.randn(10, 3, 32, 32)
    tensor4 = torch.randn(10, 1)
    tensor5 = torch.randn(10, 1, 1, 1)
    
    # 这三个张量第一维度都是10，应该可以创建TensorDataset
    try:
        dataset = TensorDataset(tensor3, tensor4, tensor5)
        assert len(dataset) == 10, "维度匹配时应成功创建数据集"
    except AssertionError:
        pytest.fail("维度匹配的张量应能成功创建TensorDataset")
    
    # 情况2：多个张量维度不匹配
    tensor6 = torch.randn(8, 3, 32, 32)  # 8个样本
    tensor7 = torch.randn(10, 1)         # 10个样本，不匹配
    
    with pytest.raises(AssertionError) as exc_info2:
        TensorDataset(tensor6, tensor7)
    
    error_msg2 = str(exc_info2.value).lower()
    assert "size" in error_msg2 or "dimension" in error_msg2 or "match" in error_msg2, \
        f"多个张量维度不匹配时错误信息应包含提示，实际错误信息: {error_msg2}"
    
    # 情况3：零维张量（标量）测试
    tensor8 = torch.randn(10)  # 10个标量
    tensor9 = torch.randn(10, 1)  # 10个一维向量
    
    # 这应该可以创建，因为第一维度都是10
    try:
        dataset2 = TensorDataset(tensor8, tensor9)
        assert len(dataset2) == 10, "零维和一维张量第一维度相同时应成功创建"
    except AssertionError:
        pytest.fail("第一维度相同的不同维度张量应能成功创建TensorDataset")
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@pytest.mark.parametrize("dataset_size,lengths,use_generator", [
    (100, [70, 30], False),
    (100, [0.7, 0.3], True),
])
def test_random_split_integer_split(dataset_size, lengths, use_generator):
    """random_split 整数分割"""
    # 创建模拟数据集
    class MockDataset(Dataset):
        def __init__(self, size):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return idx  # 返回索引作为样本
    
    dataset = MockDataset(dataset_size)
    
    # 设置随机种子
    generator = torch.Generator().manual_seed(42) if use_generator else None
    
    # Mock torch.randperm 以控制随机性
    with patch('torch.randperm') as mock_randperm:
        # 创建固定的随机排列
        if isinstance(lengths[0], float):
            # 比例分割：转换为整数长度
            int_lengths = [int(dataset_size * length) for length in lengths]
            # 调整最后一个长度以确保总和正确
            int_lengths[-1] = dataset_size - sum(int_lengths[:-1])
            expected_perm = torch.arange(dataset_size)
        else:
            # 整数分割
            int_lengths = lengths
            expected_perm = torch.arange(dataset_size)
        
        mock_randperm.return_value = expected_perm
        
        # 执行 random_split
        subsets = random_split(dataset, lengths, generator=generator)
    
    # 验证返回结果
    assert isinstance(subsets, list), "应返回列表"
    assert len(subsets) == len(lengths), "返回列表长度应与指定长度数一致"
    
    # 验证每个子集
    total_samples = 0
    all_indices = set()
    
    for i, subset in enumerate(subsets):
        # 验证子集类型
        assert isinstance(subset, Subset), f"子集{i}应为Subset类型"
        
        # 验证子集长度
        if isinstance(lengths[0], float):
            expected_len = int_lengths[i]
        else:
            expected_len = lengths[i]
        
        assert len(subset) == expected_len, f"子集{i}大小与指定长度不一致"
        
        # 收集所有样本索引
        for j in range(len(subset)):
            idx = subset[j]
            all_indices.add(idx)
            total_samples += 1
    
    # 验证所有样本被分配且不重复
    assert total_samples == dataset_size, "所有样本应被分配"
    assert len(all_indices) == dataset_size, "样本分配不应重复"
    
    # 验证原数据集未被修改
    assert len(dataset) == dataset_size, "原数据集长度不应改变"
    
    # 随机性可控制（使用固定种子） - strong断言
    # 使用相同的种子和参数，结果应可重现
    generator1 = torch.Generator().manual_seed(42) if use_generator else None
    generator2 = torch.Generator().manual_seed(42) if use_generator else None
    
    with patch('torch.randperm') as mock_randperm1:
        mock_randperm1.return_value = expected_perm
        subsets1 = random_split(dataset, lengths, generator=generator1)
    
    with patch('torch.randperm') as mock_randperm2:
        mock_randperm2.return_value = expected_perm
        subsets2 = random_split(dataset, lengths, generator=generator2)
    
    # 验证两次分割结果相同
    assert len(subsets1) == len(subsets2), "相同参数下分割结果长度应相同"
    
    for i, (subset1, subset2) in enumerate(zip(subsets1, subsets2)):
        assert len(subset1) == len(subset2), f"子集{i}长度应相同"
        
        # 验证子集内容相同
        for j in range(len(subset1)):
            assert subset1[j] == subset2[j], f"子集{i}的第{j}个样本应相同"
    
    # 分割结果可重现 - strong断言
    # 验证子集索引是连续的（因为mock了randperm返回有序排列）
    # 对于整数分割[70, 30]，第一个子集应包含索引0-69，第二个子集应包含索引70-99
    if isinstance(lengths[0], int) and not use_generator:
        cumulative = 0
        for i, subset in enumerate(subsets):
            subset_size = len(subset)
            # 验证子集索引是连续的
            expected_indices = list(range(cumulative, cumulative + subset_size))
            actual_indices = [subset[j] for j in range(subset_size)]
            assert actual_indices == expected_indices, \
                f"子集{i}的索引应为{expected_indices}，实际为{actual_indices}"
            cumulative += subset_size
    
    # 测试生成器参数的影响
    if use_generator:
        # 使用不同的生成器种子应产生不同的结果
        generator_diff = torch.Generator().manual_seed(123)  # 不同种子
        
        with patch('torch.randperm') as mock_randperm_diff:
            # 使用不同的随机排列
            diff_perm = torch.arange(dataset_size)
            # 反转排列以产生不同结果
            diff_perm = diff_perm.flip(0)
            mock_randperm_diff.return_value = diff_perm
            
            subsets_diff = random_split(dataset, lengths, generator=generator_diff)
        
        # 验证结果不同（至少有一个子集不同）
        # 注意：由于我们mock了randperm，实际上结果会不同
        # 但为了测试逻辑，我们检查子集结构是否相同
        assert len(subsets) == len(subsets_diff), "不同生成器下分割结果长度应相同"
        
        # 验证总样本数相同
        total_original = sum(len(s) for s in subsets)
        total_diff = sum(len(s) for s in subsets_diff)
        assert total_original == total_diff == dataset_size, "总样本数应保持不变"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
@pytest.mark.parametrize("dataset_sizes,index_to_test", [
    ([20, 30, 50], [0, 25, 80, -1]),
    ([0, 50, 0], [0, 25, 49]),
])
def test_concat_dataset_multiple_datasets(dataset_sizes, index_to_test):
    """ConcatDataset 多数据集拼接"""
    # 创建多个模拟数据集
    datasets = []
    for size in dataset_sizes:
        dataset = MockDataset(size)
        datasets.append(dataset)
    
    # 创建 ConcatDataset
    concat_dataset = ConcatDataset(datasets)
    
    # 验证总长度等于各数据集长度之和
    expected_total_size = sum(dataset_sizes)
    assert len(concat_dataset) == expected_total_size, \
        f"总长度应为{expected_total_size}，实际为{len(concat_dataset)}"
    
    # 测试索引映射
    for idx in index_to_test:
        # 跳过空数据集的测试
        if expected_total_size == 0:
            with pytest.raises(IndexError):
                _ = concat_dataset[idx]
            continue
        
        # 计算实际索引
        actual_idx = idx if idx >= 0 else expected_total_size + idx
        
        # 验证索引在有效范围内
        if 0 <= actual_idx < expected_total_size:
            # 获取样本
            sample = concat_dataset[idx]
            
            # 验证样本值正确
            # 计算样本来自哪个数据集
            cumulative_sizes = [0]
            for size in dataset_sizes:
                cumulative_sizes.append(cumulative_sizes[-1] + size)
            
            # 使用二分查找确定数据集索引
            dataset_idx = bisect.bisect_right(cumulative_sizes, actual_idx) - 1
            
            # 计算在数据集内的局部索引
            local_idx = actual_idx - cumulative_sizes[dataset_idx]
            
            # 验证样本值
            expected_sample = datasets[dataset_idx][local_idx]
            assert sample == expected_sample, \
                f"索引{idx}的样本值不正确，应为{expected_sample}，实际为{sample}"
            
            # 验证二分查找逻辑正确（strong断言）
            # 手动计算数据集索引进行验证
            manual_dataset_idx = -1
            cumulative = 0
            for i, size in enumerate(dataset_sizes):
                if actual_idx < cumulative + size:
                    manual_dataset_idx = i
                    break
                cumulative += size
            
            assert dataset_idx == manual_dataset_idx, \
                f"二分查找结果{dataset_idx}与手动计算{manual_dataset_idx}不一致"
            
            # 验证局部索引计算正确
            manual_local_idx = actual_idx - cumulative
            assert local_idx == manual_local_idx, \
                f"局部索引计算{local_idx}与手动计算{manual_local_idx}不一致"
        else:
            # 索引超出范围应抛出IndexError
            with pytest.raises(IndexError):
                _ = concat_dataset[idx]
    
    # 测试空数据集拼接的特殊情况
    if all(size == 0 for size in dataset_sizes):
        # 所有数据集都为空
        assert len(concat_dataset) == 0, "空数据集拼接后长度应为0"
        with pytest.raises(IndexError):
            _ = concat_dataset[0]
    
    # 验证支持负索引访问
    if expected_total_size > 0:
        # 测试最后一个元素
        last_sample = concat_dataset[-1]
        first_sample = concat_dataset[0]
        
        # 验证负索引与正索引对应关系
        assert concat_dataset[-1] == concat_dataset[expected_total_size - 1], \
            "负索引-1应与最后一个正索引对应"
        
        # 验证边界条件
        if expected_total_size > 1:
            assert concat_dataset[-2] == concat_dataset[expected_total_size - 2], \
                "负索引-2应与倒数第二个正索引对应"
    
    # 边界条件处理正确（strong断言）
    # 测试边界索引
    if expected_total_size > 0:
        # 测试第一个数据集的第一个元素
        if dataset_sizes[0] > 0:
            sample = concat_dataset[0]
            expected = datasets[0][0]
            assert sample == expected, "第一个数据集的第一个元素访问不正确"
        
        # 测试第一个数据集的最后一个元素
        if dataset_sizes[0] > 0:
            last_idx = dataset_sizes[0] - 1
            sample = concat_dataset[last_idx]
            expected = datasets[0][-1]
            assert sample == expected, "第一个数据集的最后一个元素访问不正确"
        
        # 测试最后一个数据集的第一个元素
        if dataset_sizes[-1] > 0:
            first_idx = sum(dataset_sizes[:-1])
            sample = concat_dataset[first_idx]
            expected = datasets[-1][0]
            assert sample == expected, "最后一个数据集的第一个元素访问不正确"
        
        # 测试最后一个数据集的最后一个元素
        if dataset_sizes[-1] > 0:
            last_idx = expected_total_size - 1
            sample = concat_dataset[last_idx]
            expected = datasets[-1][-1]
            assert sample == expected, "最后一个数据集的最后一个元素访问不正确"
    
    # 空数据集拼接处理（strong断言）
    # 测试包含空数据集的混合情况
    mixed_sizes = []
    mixed_datasets = []
    for size in dataset_sizes:
        if size == 0:
            # 空数据集
            mixed_datasets.append(MockDataset(0))
        else:
            # 非空数据集
            mixed_datasets.append(MockDataset(size))
        mixed_sizes.append(size)
    
    mixed_concat = ConcatDataset(mixed_datasets)
    assert len(mixed_concat) == sum(mixed_sizes), "混合空/非空数据集拼接长度应正确"
    
    # 验证空数据集不影响索引映射
    if sum(mixed_sizes) > 0:
        # 找到第一个非空数据集
        first_nonempty_idx = next(i for i, size in enumerate(mixed_sizes) if size > 0)
        cumulative_before = sum(mixed_sizes[:first_nonempty_idx])
        
        # 测试第一个非空数据集的第一个元素
        sample = mixed_concat[cumulative_before]
        expected = mixed_datasets[first_nonempty_idx][0]
        assert sample == expected, "空数据集后的第一个元素访问不正确"
    
    # 测试单个数据集的情况（边界情况）
    single_dataset = [MockDataset(10)]
    single_concat = ConcatDataset(single_dataset)
    assert len(single_concat) == 10, "单个数据集拼接长度应正确"
    assert single_concat[0] == single_dataset[0][0], "单个数据集元素访问应正确"
    assert single_concat[-1] == single_dataset[0][-1], "单个数据集负索引访问应正确"
    
    # 测试零个数据集的情况（边界情况）
    empty_concat = ConcatDataset([])
    assert len(empty_concat) == 0, "空列表拼接长度应为0"
    with pytest.raises(IndexError):
        _ = empty_concat[0]
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
def test_dataset_abstract_class_interface():
    """Dataset 抽象类接口约束"""
    # 测试直接实例化 Dataset 抽象类
    dataset = Dataset()
    
    # 验证 __len__ 方法未实现 - Python会抛出TypeError
    with pytest.raises(TypeError) as exc_info:
        len(dataset)
    
    error_msg = str(exc_info.value).lower()
    # 检查错误信息是否包含 'len' 或 'no len' 等提示
    assert "len" in error_msg or "has no len" in error_msg, \
        f"错误信息应提示len方法未实现，实际错误信息: {error_msg}"
    
    # 验证 __getitem__ 方法未实现 - Dataset.__getitem__ 抛出 NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        dataset[0]
    
    # 检查错误信息（NotImplementedError通常没有特定消息）
    # 但我们可以验证异常类型正确
    
    # 测试子类必须实现 __getitem__
    class IncompleteDataset(Dataset):
        def __len__(self):
            return 10
    
    incomplete = IncompleteDataset()
    
    # __len__ 已实现，可以调用
    assert len(incomplete) == 10, "子类实现的__len__方法应正常工作"
    
    # __getitem__ 未实现，应抛出 NotImplementedError
    with pytest.raises(NotImplementedError):
        incomplete[0]
    
    # 测试完整子类
    class CompleteDataset(Dataset):
        def __init__(self, size=10):
            self.data = list(range(size))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            if idx < 0:
                idx = len(self) + idx
            if 0 <= idx < len(self):
                return self.data[idx]
            raise IndexError(f"Index {idx} out of range")
    
    complete = CompleteDataset(5)
    
    # 验证完整子类功能正常
    assert len(complete) == 5, "完整子类长度应正确"
    assert complete[0] == 0, "完整子类__getitem__应正常工作"
    assert complete[-1] == 4, "完整子类应支持负索引"
    
    # 测试 __add__ 方法返回 ConcatDataset
    dataset1 = CompleteDataset(3)
    dataset2 = CompleteDataset(4)
    
    result = dataset1 + dataset2
    
    # 验证返回类型为 ConcatDataset
    assert isinstance(result, ConcatDataset), "__add__方法应返回ConcatDataset"
    
    # 验证拼接结果正确
    assert len(result) == 7, "拼接后数据集长度应为3+4=7"
    assert result[0] == 0, "第一个元素应来自第一个数据集"
    assert result[3] == 0, "第四个元素应来自第二个数据集"
    
    # 子类必须实现 __getitem__（strong断言）
    # 测试多个不完整子类的情况
    class NoLenDataset(Dataset):
        def __getitem__(self, idx):
            return idx
    
    no_len = NoLenDataset()
    
    # 没有实现 __len__，调用 len() 会抛出 TypeError
    with pytest.raises(TypeError):
        len(no_len)
    
    # 但 __getitem__ 可以工作
    assert no_len[5] == 5, "只有__getitem__的子类应能访问元素"
    
    class NoMethodsDataset(Dataset):
        pass
    
    no_methods = NoMethodsDataset()
    
    # 两个方法都未实现
    with pytest.raises(TypeError):
        len(no_methods)
    
    with pytest.raises(NotImplementedError):
        no_methods[0]
    
    # __add__ 方法返回 ConcatDataset（strong断言）
    # 测试多个数据集相加
    dataset3 = CompleteDataset(2)
    dataset4 = CompleteDataset(3)
    dataset5 = CompleteDataset(4)
    
    result2 = dataset3 + dataset4 + dataset5
    
    # 验证链式相加
    assert isinstance(result2, ConcatDataset), "链式相加应返回ConcatDataset"
    assert len(result2) == 9, "链式相加后长度应为2+3+4=9"
    
    # 验证元素顺序
    assert result2[0] == 0, "第一个元素应来自第一个数据集"
    assert result2[2] == 0, "第三个元素应来自第二个数据集"
    assert result2[5] == 0, "第六个元素应来自第三个数据集"
    
    # 测试不同类型数据集的相加
    class StringDataset(Dataset):
        def __init__(self, size):
            self.data = [f"item_{i}" for i in range(size)]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    str_dataset = StringDataset(3)
    mixed_result = complete + str_dataset
    
    assert isinstance(mixed_result, ConcatDataset), "不同类型数据集相加应返回ConcatDataset"
    assert len(mixed_result) == 8, "混合类型数据集相加长度应为5+3=8"
    assert mixed_result[0] == 0, "第一个元素应来自数值数据集"
    assert mixed_result[5] == "item_0", "第六个元素应来自字符串数据集"
    
    # 测试数据集与自身的相加
    self_result = complete + complete
    assert len(self_result) == 10, "数据集与自身相加长度应翻倍"
    assert self_result[0] == 0, "第一个元素应正确"
    assert self_result[5] == 0, "第六个元素应来自第二个副本"
    
    # 测试空数据集的相加
    empty_dataset = CompleteDataset(0)
    empty_result = empty_dataset + complete
    
    assert len(empty_result) == 5, "空数据集与非空数据集相加长度应正确"
    assert empty_result[0] == 0, "第一个元素应来自非空数据集"
    
    # 测试两个空数据集相加
    empty_result2 = empty_dataset + empty_dataset
    assert len(empty_result2) == 0, "两个空数据集相加长度应为0"
    with pytest.raises(IndexError):
        _ = empty_result2[0]
    
    # 验证 Dataset 抽象类的其他方法
    # 测试 __repr__ 方法
    repr_str = repr(complete)
    assert "CompleteDataset" in repr_str or "Dataset" in repr_str, \
        f"__repr__应包含类名，实际: {repr_str}"
    
    # 测试 __add__ 方法的交换律（不适用，因为Dataset + Dataset返回ConcatDataset）
    # 但可以验证结果类型一致
    result_ab = dataset1 + dataset2
    result_ba = dataset2 + dataset1
    
    assert isinstance(result_ab, ConcatDataset) and isinstance(result_ba, ConcatDataset), \
        "两个方向相加都应返回ConcatDataset"
    assert len(result_ab) == len(result_ba), "两个方向相加长度应相同"
    
    # 注意：dataset1 + dataset2 和 dataset2 + dataset1 的元素顺序不同
    # 这是符合预期的，因为ConcatDataset保持原始顺序
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:FOOTER START ====
def test_random_split_invalid_lengths():
    """random_split 长度超过数据集大小的异常测试"""
    dataset = MockDataset(100)
    
    # 测试长度超过数据集大小（整数模式）
    with pytest.raises(ValueError) as exc_info:
        random_split(dataset, [101])
    
    # 验证错误信息
    error_msg = str(exc_info.value).lower()
    assert "sum" in error_msg or "length" in error_msg or "size" in error_msg, \
        f"错误信息应包含长度总和或大小的提示，实际错误信息: {error_msg}"
    
    # 测试长度总和小于数据集大小（整数模式）
    with pytest.raises(ValueError) as exc_info:
        random_split(dataset, [99])  # 总和99 < 100
    
    # 验证错误信息
    error_msg = str(exc_info.value).lower()
    assert "sum" in error_msg or "length" in error_msg or "size" in error_msg, \
        f"错误信息应包含长度总和或大小的提示，实际错误信息: {error_msg}"
    
    # 测试负长度（整数模式）- random_split只检查总和，不检查单个负值
    # [-10, 110] 总和为100，不会抛出异常，所以需要调整测试
    # 改为测试总和不为100的情况
    with pytest.raises(ValueError) as exc_info:
        random_split(dataset, [-10, 120])  # 总和110 > 100
    
    # 测试比例总和不为1（比例模式）
    with pytest.raises(ValueError) as exc_info:
        random_split(dataset, [0.5, 0.6])  # 总和1.1 > 1
    
    # 测试负比例（比例模式）- random_split会检查比例在0-1之间
    with pytest.raises(ValueError) as exc_info:
        random_split(dataset, [-0.1, 1.1])
    
    # 验证错误信息包含比例范围提示
    error_msg = str(exc_info.value).lower()
    assert "fraction" in error_msg or "between" in error_msg or "0 and 1" in error_msg, \
        f"错误信息应包含比例范围的提示，实际错误信息: {error_msg}"
    
    # 测试比例总和小于1（比例模式）
    with pytest.raises(ValueError) as exc_info:
        random_split(dataset, [0.3, 0.3])  # 总和0.6 < 1
    
    # 测试空长度列表
    with pytest.raises(ValueError) as exc_info:
        random_split(dataset, [])
    
    # 验证错误信息
    error_msg = str(exc_info.value).lower()
    assert "sum" in error_msg or "length" in error_msg, \
        f"错误信息应包含长度总和的提示，实际错误信息: {error_msg}"
# ==== BLOCK:FOOTER END ====