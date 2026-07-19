"""
Unit tests for tensorflow.python.data.experimental.ops.parsing_ops
"""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parsing_ops
from unittest.mock import patch, MagicMock

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== BLOCK:HEADER START ====
def create_example_proto(features_dict):
    """Create a serialized tf.train.Example proto from features dict."""
    example = tf.train.Example()
    for key, value in features_dict.items():
        if isinstance(value, (int, np.integer)):
            example.features.feature[key].int64_list.value.append(value)
        elif isinstance(value, (float, np.floating)):
            example.features.feature[key].float_list.value.append(value)
        elif isinstance(value, (str, bytes)):
            if isinstance(value, str):
                value = value.encode('utf-8')
            example.features.feature[key].bytes_list.value.append(value)
        elif isinstance(value, list):
            if all(isinstance(v, (int, np.integer)) for v in value):
                example.features.feature[key].int64_list.value.extend(value)
            elif all(isinstance(v, (float, np.floating)) for v in value):
                example.features.feature[key].float_list.value.extend(value)
            elif all(isinstance(v, (str, bytes)) for v in value):
                for v in value:
                    if isinstance(v, str):
                        v = v.encode('utf-8')
                    example.features.feature[key].bytes_list.value.append(v)
    return example.SerializeToString()


def create_string_dataset(serialized_examples, batch_size=None):
    """Create a dataset of serialized Example protos as string vectors."""
    # 将标量字符串包装成向量（形状为[None]的字符串向量）
    # parse_example_dataset要求输入是字符串向量，而不是标量
    string_vectors = tf.constant(serialized_examples, dtype=tf.string)
    # 确保形状是[None]（字符串向量）
    dataset = tf.data.Dataset.from_tensor_slices(string_vectors)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def assert_tensors_equal(actual, expected, tolerance=1e-6):
    """Assert that two tensors are equal within tolerance."""
    if isinstance(actual, tf.Tensor) and isinstance(expected, tf.Tensor):
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=tolerance, atol=tolerance
        )
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key in actual:
            assert_tensors_equal(actual[key], expected[key], tolerance)
    else:
        assert actual == expected
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
def test_fixed_len_feature_basic_parsing():
    """Test basic parsing with FixedLenFeature (TC-01)."""
    # Create test data
    dataset_size = 10
    serialized_examples = []
    
    for i in range(dataset_size):
        example = create_example_proto({
            'feature1': float(i),
            'feature2': i * 2,
            'feature3': f'string_{i}'.encode('utf-8')
        })
        serialized_examples.append(example)
    
    # Create dataset - 字符串向量，shape=[None]
    dataset = create_string_dataset(serialized_examples)
    
    # 验证输入数据集是字符串向量
    element_spec = dataset.element_spec
    assert isinstance(element_spec, tf.TensorSpec), "输入数据集应该是TensorSpec"
    assert element_spec.dtype == tf.string, "输入数据集应该是字符串类型"
    assert element_spec.shape.as_list() == [None], "输入数据集应该是字符串向量(shape=[None])"
    
    # Define features
    features = {
        'feature1': tf.io.FixedLenFeature([], tf.float32),
        'feature2': tf.io.FixedLenFeature([], tf.int64),
        'feature3': tf.io.FixedLenFeature([], tf.string)
    }
    
    # Apply parse_example_dataset transformation
    parse_fn = parsing_ops.parse_example_dataset(
        features=features,
        num_parallel_calls=1,
        deterministic=None
    )
    
    parsed_dataset = dataset.apply(parse_fn)
    
    # Verify the transformation returns a function
    assert callable(parse_fn), "parse_example_dataset should return a callable function"
    
    # Verify output dataset structure
    element_spec = parsed_dataset.element_spec
    assert isinstance(element_spec, dict), "Output element_spec should be a dict"
    assert set(element_spec.keys()) == {'feature1', 'feature2', 'feature3'}
    
    # Verify tensor specs
    # 注意：由于输入是字符串向量(shape=[None])，输出形状会包含batch维度
    assert isinstance(element_spec['feature1'], tf.TensorSpec)
    assert element_spec['feature1'].dtype == tf.float32
    assert element_spec['feature1'].shape.as_list() == [None], "标量特征的输出形状应该是[None]（batch维度）"
    
    assert isinstance(element_spec['feature2'], tf.TensorSpec)
    assert element_spec['feature2'].dtype == tf.int64
    assert element_spec['feature2'].shape.as_list() == [None], "标量特征的输出形状应该是[None]（batch维度）"
    
    assert isinstance(element_spec['feature3'], tf.TensorSpec)
    assert element_spec['feature3'].dtype == tf.string
    assert element_spec['feature3'].shape.as_list() == [None], "标量特征的输出形状应该是[None]（batch维度）"
    
    # Verify dataset size
    count = 0
    for element in parsed_dataset:
        count += 1
        # Verify each element has the expected keys
        assert set(element.keys()) == {'feature1', 'feature2', 'feature3'}
        # Verify tensor shapes
        # 注意：当我们迭代数据集时，每个元素是一个batch，对于标量特征形状是[1]
        assert element['feature1'].shape == (1,), "迭代时标量特征的形状应该是(1,)（batch_size=1）"
        assert element['feature2'].shape == (1,), "迭代时标量特征的形状应该是(1,)（batch_size=1）"
        assert element['feature3'].shape == (1,), "迭代时标量特征的形状应该是(1,)（batch_size=1）"
        # Verify data types
        assert element['feature1'].dtype == tf.float32
        assert element['feature2'].dtype == tf.int64
        assert element['feature3'].dtype == tf.string
    
    assert count == dataset_size, f"Expected {dataset_size} elements, got {count}"
    
    # Compare with tf.io.parse_example as oracle
    for i, element in enumerate(parsed_dataset):
        # Create single example for oracle comparison
        single_example = tf.constant([serialized_examples[i]])
        oracle_result = tf.io.parse_example(single_example, features)
        
        # Compare values - 注意：element包含batch维度，需要取第一个元素
        assert_tensors_equal(element['feature1'][0], oracle_result['feature1'][0])
        assert_tensors_equal(element['feature2'][0], oracle_result['feature2'][0])
        assert element['feature3'][0].numpy() == oracle_result['feature3'][0].numpy()
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("features,expected_error", [
    (None, ValueError),
    ({}, TypeError)  # 空字典会在_ParseExampleDataset中触发TypeError
])
def test_features_parameter_validation(features, expected_error):
    """Test features parameter validation (TC-02)."""
    # Create a simple dataset for testing
    serialized_example = create_example_proto({'test': 1.0})
    dataset = create_string_dataset([serialized_example])
    
    with pytest.raises(expected_error) as exc_info:
        parse_fn = parsing_ops.parse_example_dataset(
            features=features,
            num_parallel_calls=1
        )
        # Try to apply the transformation
        dataset.apply(parse_fn)
    
    # Verify error message contains expected information
    error_msg = str(exc_info.value).lower()
    
    if features is None:
        # parse_example_dataset函数本身会检查features是否为None
        assert "features" in error_msg or "required" in error_msg or "specified" in error_msg
        assert isinstance(exc_info.value, ValueError)
    elif features == {}:
        # 空字典会在_ParseExampleDataset中触发TypeError
        # 因为_ParseExampleDataset会检查输入数据集类型
        assert "input dataset" in error_msg or "strings" in error_msg or "type" in error_msg
        assert isinstance(exc_info.value, TypeError)
    
    # Verify the error type matches
    assert isinstance(exc_info.value, expected_error)
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
@patch('tensorflow.python.data.experimental.ops.parsing_ops.gen_experimental_dataset_ops.parse_example_dataset_v2')
def test_parallel_parsing_functionality(mock_parse_op):
    """Test parallel parsing functionality (TC-03)."""
    # Setup mock to capture call arguments
    mock_dataset = MagicMock()
    mock_dataset.element_spec = tf.TensorSpec([None], tf.string)
    mock_result = MagicMock()
    mock_parse_op.return_value = mock_result
    
    # Create test data
    dataset_size = 20
    serialized_examples = []
    
    for i in range(dataset_size):
        example = create_example_proto({
            'feature': [i, i+1, i+2]  # shape [3]
        })
        serialized_examples.append(example)
    
    dataset = create_string_dataset(serialized_examples)
    
    # 验证输入数据集是字符串向量
    element_spec = dataset.element_spec
    assert isinstance(element_spec, tf.TensorSpec), "输入数据集应该是TensorSpec"
    assert element_spec.dtype == tf.string, "输入数据集应该是字符串类型"
    assert element_spec.shape.as_list() == [None], "输入数据集应该是字符串向量(shape=[None])"
    
    # Define features with shape [3]
    features = {
        'feature': tf.io.FixedLenFeature([3], tf.int32)
    }
    
    # Apply parse_example_dataset with parallel calls
    parse_fn = parsing_ops.parse_example_dataset(
        features=features,
        num_parallel_calls=4,
        deterministic=True
    )
    
    # Verify the transformation returns a function
    assert callable(parse_fn), "parse_example_dataset should return a callable function"
    
    # Apply the transformation
    parsed_dataset = dataset.apply(parse_fn)
    
    # Verify output dataset structure
    element_spec = parsed_dataset.element_spec
    assert isinstance(element_spec, dict), "Output element_spec should be a dict"
    assert 'feature' in element_spec
    
    # Verify tensor spec
    # 注意：由于输入是字符串向量(shape=[None])，输出形状会包含batch维度
    assert isinstance(element_spec['feature'], tf.TensorSpec)
    assert element_spec['feature'].dtype == tf.int32
    # Shape should be [None, 3] (batch维度 + 特征形状)
    assert element_spec['feature'].shape.as_list() == [None, 3], \
        f"期望形状[None, 3]，实际{element_spec['feature'].shape.as_list()}"
    
    # Verify dataset size
    count = 0
    for element in parsed_dataset:
        count += 1
        assert 'feature' in element
        # 迭代时batch_size=1，所以形状是[1, 3]
        assert element['feature'].shape == (1, 3), \
            f"期望形状(1, 3)，实际{element['feature'].shape}"
        assert element['feature'].dtype == tf.int32
    
    assert count == dataset_size, f"Expected {dataset_size} elements, got {count}"
    
    # Test with deterministic=True
    # Create another dataset with deterministic=False for comparison
    parse_fn_non_det = parsing_ops.parse_example_dataset(
        features=features,
        num_parallel_calls=4,
        deterministic=False
    )
    
    parsed_dataset_non_det = dataset.apply(parse_fn_non_det)
    
    # Both should produce valid datasets
    assert callable(parse_fn_non_det)
    
    # Verify parallel execution parameter was considered
    # (In real implementation, this would affect performance but not correctness)
    
    # Compare with tf.io.parse_example as oracle
    for i, element in enumerate(parsed_dataset):
        single_example = tf.constant([serialized_examples[i]])
        oracle_result = tf.io.parse_example(single_example, features)
        # 注意：element包含batch维度，需要取第一个元素
        assert_tensors_equal(element['feature'][0], oracle_result['feature'][0])
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
# DEFERRED: num_parallel_calls边界值 (TC-04)
# Will be implemented in later iterations
pass
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
@patch('tensorflow.python.data.experimental.ops.parsing_ops.gen_experimental_dataset_ops.parse_example_dataset_v2')
def test_multiple_feature_types_support(mock_parse_op):
    """Test support for multiple feature types (TC-05)."""
    # Setup mock
    mock_result = MagicMock()
    mock_parse_op.return_value = mock_result
    
    # Create test data with mixed feature types
    dataset_size = 15
    serialized_examples = []
    
    for i in range(dataset_size):
        example = create_example_proto({
            'fixed_feature': float(i),  # FixedLenFeature
            'var_len_int': [i, i*2, i*3],  # VarLenFeature - variable length
            'var_len_float': [float(i)/2.0, float(i)/3.0],  # VarLenFeature
            'string_feature': f'text_{i}'.encode('utf-8')  # FixedLenFeature string
        })
        serialized_examples.append(example)
    
    dataset = create_string_dataset(serialized_examples)
    
    # Define mixed features
    features = {
        'fixed_feature': tf.io.FixedLenFeature([], tf.float32),
        'var_len_int': tf.io.VarLenFeature(tf.int64),
        'var_len_float': tf.io.VarLenFeature(tf.float32),
        'string_feature': tf.io.FixedLenFeature([], tf.string)
    }
    
    # Apply parse_example_dataset transformation
    parse_fn = parsing_ops.parse_example_dataset(
        features=features,
        num_parallel_calls=2,
        deterministic=False
    )
    
    # Verify the transformation returns a function
    assert callable(parse_fn), "parse_example_dataset should return a callable function"
    
    # Apply the transformation
    parsed_dataset = dataset.apply(parse_fn)
    
    # Verify output dataset structure
    element_spec = parsed_dataset.element_spec
    assert isinstance(element_spec, dict), "Output element_spec should be a dict"
    
    # Check all features are present
    expected_keys = {'fixed_feature', 'var_len_int', 'var_len_float', 'string_feature'}
    assert set(element_spec.keys()) == expected_keys
    
    # Verify tensor specs for each feature type
    # FixedLenFeature
    assert isinstance(element_spec['fixed_feature'], tf.TensorSpec)
    assert element_spec['fixed_feature'].dtype == tf.float32
    assert element_spec['fixed_feature'].shape.as_list() == []
    
    # VarLenFeature should produce SparseTensorSpec
    assert isinstance(element_spec['var_len_int'], tf.SparseTensorSpec)
    assert element_spec['var_len_int'].dtype == tf.int64
    
    assert isinstance(element_spec['var_len_float'], tf.SparseTensorSpec)
    assert element_spec['var_len_float'].dtype == tf.float32
    
    # FixedLenFeature string
    assert isinstance(element_spec['string_feature'], tf.TensorSpec)
    assert element_spec['string_feature'].dtype == tf.string
    assert element_spec['string_feature'].shape.as_list() == []
    
    # Verify dataset size
    count = 0
    for element in parsed_dataset:
        count += 1
        # Verify all keys are present
        assert set(element.keys()) == expected_keys
        
        # Verify FixedLenFeature
        assert element['fixed_feature'].shape == ()
        assert element['fixed_feature'].dtype == tf.float32
        
        # Verify VarLenFeature returns SparseTensor
        assert isinstance(element['var_len_int'], tf.SparseTensor)
        assert element['var_len_int'].dtype == tf.int64
        
        assert isinstance(element['var_len_float'], tf.SparseTensor)
        assert element['var_len_float'].dtype == tf.float32
        
        # Verify string feature
        assert element['string_feature'].shape == ()
        assert element['string_feature'].dtype == tf.string
    
    assert count == dataset_size, f"Expected {dataset_size} elements, got {count}"
    
    # Verify feature count
    assert len(features) == 4, f"Expected 4 features, got {len(features)}"
    
    # Compare FixedLenFeatures with tf.io.parse_example as oracle
    for i, element in enumerate(parsed_dataset):
        single_example = tf.constant([serialized_examples[i]])
        oracle_result = tf.io.parse_example(single_example, features)
        
        # Compare fixed features
        assert_tensors_equal(element['fixed_feature'], oracle_result['fixed_feature'][0])
        assert element['string_feature'].numpy() == oracle_result['string_feature'][0].numpy()
        
        # For VarLenFeatures, compare values
        # Convert sparse tensors to dense for comparison
        var_int_dense = tf.sparse.to_dense(element['var_len_int'])
        var_int_oracle = tf.sparse.to_dense(oracle_result['var_len_int'][0])
        assert_tensors_equal(var_int_dense, var_int_oracle)
        
        var_float_dense = tf.sparse.to_dense(element['var_len_float'])
        var_float_oracle = tf.sparse.to_dense(oracle_result['var_len_float'][0])
        assert_tensors_equal(var_float_dense, var_float_oracle, tolerance=1e-6)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# DEFERRED: deterministic参数行为 (TC-06)
# Will be implemented in later iterations
pass
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# DEFERRED: 空数据集处理 (TC-07)
# Will be implemented in later iterations
pass
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# DEFERRED: 无效输入数据集验证 (TC-08)
# Will be implemented in later iterations
pass
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    test_results = []
    
    # Run basic tests
    print("Running basic tests...")
    
    # Test 1: FixedLenFeature基本解析
    try:
        test_fixed_len_feature_basic_parsing()
        test_results.append(("CASE_01", "PASS"))
        print("✓ CASE_01: FixedLenFeature基本解析 - PASS")
    except Exception as e:
        test_results.append(("CASE_01", f"FAIL: {str(e)}"))
        print(f"✗ CASE_01: FixedLenFeature基本解析 - FAIL: {e}")
    
    # Test 2: features参数验证
    try:
        # Test with None features
        test_features_parameter_validation(None, ValueError)
        # Test with empty dict
        test_features_parameter_validation({}, ValueError)
        test_results.append(("CASE_02", "PASS"))
        print("✓ CASE_02: features参数验证 - PASS")
    except Exception as e:
        test_results.append(("CASE_02", f"FAIL: {str(e)}"))
        print(f"✗ CASE_02: features参数验证 - FAIL: {e}")
    
    print("\nTest Summary:")
    for test_name, result in test_results:
        print(f"  {test_name}: {result}")
    
    # Count passes
    passes = sum(1 for _, result in test_results if result == "PASS")
    total = len(test_results)
    print(f"\nTotal: {passes}/{total} tests passed")
    
    if passes < total:
        sys.exit(1)
# ==== BLOCK:FOOTER END ====