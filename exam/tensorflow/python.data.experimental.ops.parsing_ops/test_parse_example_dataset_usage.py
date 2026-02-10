"""Test to understand parse_example_dataset usage"""
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parsing_ops

# 创建一些序列化的Example protos
def create_example_proto(value):
    example = tf.train.Example()
    example.features.feature['feature'].float_list.value.append(float(value))
    return example.SerializeToString()

# 创建测试数据
serialized_examples = [create_example_proto(i) for i in range(5)]

print("原始序列化示例:", serialized_examples)

# 方法1：创建标量字符串数据集
dataset1 = tf.data.Dataset.from_tensor_slices(serialized_examples)
print("\n方法1 - 标量字符串数据集:")
print("element_spec:", dataset1.element_spec)
print("形状:", dataset1.element_spec.shape)

# 方法2：创建字符串向量数据集（每个元素是一个单元素向量）
# 将每个标量字符串包装成单元素列表
string_vectors = [[example] for example in serialized_examples]
dataset2 = tf.data.Dataset.from_tensor_slices(string_vectors)
# 确保数据类型是tf.string
dataset2 = dataset2.map(lambda x: tf.constant(x, dtype=tf.string))
print("\n方法2 - 字符串向量数据集（单元素向量）:")
print("element_spec:", dataset2.element_spec)
print("形状:", dataset2.element_spec.shape)

# 方法3：创建字符串向量数据集（批处理）
dataset3 = tf.data.Dataset.from_tensor_slices(serialized_examples)
dataset3 = dataset3.batch(2)  # 批量大小为2
print("\n方法3 - 批处理字符串数据集（批量大小=2）:")
print("element_spec:", dataset3.element_spec)
print("形状:", dataset3.element_spec.shape)

# 测试parse_example_dataset
features = {'feature': tf.io.FixedLenFeature([], tf.float32)}

print("\n测试parse_example_dataset:")
for i, dataset in enumerate([dataset1, dataset2, dataset3], 1):
    print(f"\n数据集{i}:")
    try:
        parse_fn = parsing_ops.parse_example_dataset(features=features)
        parsed_dataset = dataset.apply(parse_fn)
        print(f"  成功: element_spec = {parsed_dataset.element_spec}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")