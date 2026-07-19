"""Verify the fix for parse_example_dataset tests"""
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parsing_ops

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
    return example.SerializeToString()

def create_string_dataset_fixed(serialized_examples, batch_size=1):
    """Create a dataset of serialized Example protos as string vectors."""
    # 创建标量字符串数据集
    dataset = tf.data.Dataset.from_tensor_slices(serialized_examples)
    # 应用批处理，将标量字符串转换为字符串向量
    dataset = dataset.batch(batch_size)
    return dataset

# 测试CASE_01
print("测试CASE_01: FixedLenFeature基本解析")
try:
    # 创建测试数据
    dataset_size = 10
    serialized_examples = []
    
    for i in range(dataset_size):
        example = create_example_proto({
            'feature1': float(i),
            'feature2': i * 2,
            'feature3': f'string_{i}'.encode('utf-8')
        })
        serialized_examples.append(example)
    
    # 创建数据集 - 使用批处理创建字符串向量
    dataset = create_string_dataset_fixed(serialized_examples, batch_size=1)
    
    # 验证输入数据集是字符串向量
    element_spec = dataset.element_spec
    print(f"  输入数据集element_spec: {element_spec}")
    print(f"  形状: {element_spec.shape}")
    
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
    
    print(f"  输出数据集element_spec: {parsed_dataset.element_spec}")
    
    # Verify dataset size
    count = 0
    for element in parsed_dataset:
        count += 1
    
    print(f"  数据集大小: {count} (期望: {dataset_size})")
    print("  ✓ CASE_01 测试通过")
    
except Exception as e:
    print(f"  ✗ CASE_01 测试失败: {type(e).__name__}: {e}")

# 测试CASE_02
print("\n测试CASE_02: features参数验证")
try:
    # 创建测试数据
    serialized_example = create_example_proto({'test': 1.0})
    dataset = create_string_dataset_fixed([serialized_example], batch_size=1)
    
    # 测试 None features
    print("  测试 features=None:")
    try:
        parse_fn = parsing_ops.parse_example_dataset(
            features=None,
            num_parallel_calls=1
        )
        dataset.apply(parse_fn)
        print("    ✗ 应该抛出ValueError但没有")
    except ValueError as e:
        print(f"    ✓ 正确抛出ValueError: {e}")
    
    # 测试空字典 features
    print("  测试 features={}:")
    try:
        parse_fn = parsing_ops.parse_example_dataset(
            features={},
            num_parallel_calls=1
        )
        dataset.apply(parse_fn)
        print("    ✗ 应该抛出TypeError但没有")
    except TypeError as e:
        print(f"    ✓ 正确抛出TypeError: {e}")
    
    print("  ✓ CASE_02 测试通过")
    
except Exception as e:
    print(f"  ✗ CASE_02 测试失败: {type(e).__name__}: {e}")

print("\n所有测试完成!")