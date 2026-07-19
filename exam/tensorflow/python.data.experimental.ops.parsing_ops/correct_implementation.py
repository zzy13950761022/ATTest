"""Correct implementation of parse_example_dataset tests"""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parsing_ops

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

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
    return example.SerializeToString()

def create_string_dataset(serialized_examples, batch_size=1):
    """Create a dataset of serialized Example protos as string vectors.
    
    parse_example_dataset requires input to be a dataset of string vectors
    (shape=[None]), not scalar strings. We use batch() to create string vectors.
    """
    # 创建标量字符串数据集
    dataset = tf.data.Dataset.from_tensor_slices(serialized_examples)
    # 应用批处理，将标量字符串转换为字符串向量
    dataset = dataset.batch(batch_size)
    return dataset

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
    
    # Create dataset - 使用批处理创建字符串向量
    dataset = create_string_dataset(serialized_examples, batch_size=1)
    
    # 验证输入数据集是字符串向量
    element_spec = dataset.element_spec
    assert isinstance(element_spec, tf.TensorSpec), "输入数据集应该是TensorSpec"
    assert element_spec.dtype == tf.string, "输入数据集应该是字符串类型"
    # 由于batch_size=1，形状应该是[1]（字符串向量）
    assert element_spec.shape.as_list() == [1], f"输入数据集应该是字符串向量(shape=[1])，实际{element_spec.shape}"
    
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
    
    # Verify dataset size
    count = 0
    for element in parsed_dataset:
        count += 1
        # Verify each element has the expected keys
        assert set(element.keys()) == {'feature1', 'feature2', 'feature3'}
    
    assert count == dataset_size, f"Expected {dataset_size} elements, got {count}"
    
    print("✓ test_fixed_len_feature_basic_parsing passed")

@pytest.mark.parametrize("features,expected_error", [
    (None, ValueError),
    ({}, TypeError)  # 空字典会在_ParseExampleDataset中触发TypeError
])
def test_features_parameter_validation(features, expected_error):
    """Test features parameter validation (TC-02)."""
    # Create a simple dataset for testing
    serialized_example = create_example_proto({'test': 1.0})
    dataset = create_string_dataset([serialized_example], batch_size=1)
    
    with pytest.raises(expected_error) as exc_info:
        parse_fn = parsing_ops.parse_example_dataset(
            features=features,
            num_parallel_calls=1
        )
        # Try to apply the transformation
        dataset.apply(parse_fn)
    
    # Verify the error type matches
    assert isinstance(exc_info.value, expected_error)
    
    print(f"✓ test_features_parameter_validation passed for features={features}")

def test_parallel_parsing_functionality():
    """Test parallel parsing functionality (TC-03)."""
    # Create test data
    dataset_size = 20
    serialized_examples = []
    
    for i in range(dataset_size):
        example = create_example_proto({
            'feature': [i, i+1, i+2]  # shape [3]
        })
        serialized_examples.append(example)
    
    dataset = create_string_dataset(serialized_examples, batch_size=1)
    
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
    
    # Verify dataset size
    count = 0
    for element in parsed_dataset:
        count += 1
    
    assert count == dataset_size, f"Expected {dataset_size} elements, got {count}"
    
    print("✓ test_parallel_parsing_functionality passed")

if __name__ == "__main__":
    print("运行测试...")
    try:
        test_fixed_len_feature_basic_parsing()
    except Exception as e:
        print(f"✗ test_fixed_len_feature_basic_parsing failed: {e}")
    
    try:
        test_features_parameter_validation(None, ValueError)
        test_features_parameter_validation({}, TypeError)
    except Exception as e:
        print(f"✗ test_features_parameter_validation failed: {e}")
    
    try:
        test_parallel_parsing_functionality()
    except Exception as e:
        print(f"✗ test_parallel_parsing_functionality failed: {e}")
    
    print("\n测试完成!")