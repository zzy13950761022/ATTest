import tensorflow as tf

# 测试 from_tensor_slices 的行为
examples = ["example1", "example2", "example3"]

# 方法1：直接使用列表
dataset1 = tf.data.Dataset.from_tensor_slices(examples)
print("方法1 - 直接使用列表:")
print(f"  element_spec: {dataset1.element_spec}")
print(f"  shape: {dataset1.element_spec.shape}")
print(f"  shape.as_list(): {dataset1.element_spec.shape.as_list()}")

# 方法2：使用 tf.constant
string_tensor = tf.constant(examples, dtype=tf.string)
print("\n方法2 - 使用 tf.constant:")
print(f"  string_tensor shape: {string_tensor.shape}")
dataset2 = tf.data.Dataset.from_tensor_slices(string_tensor)
print(f"  element_spec: {dataset2.element_spec}")
print(f"  shape: {dataset2.element_spec.shape}")
print(f"  shape.as_list(): {dataset2.element_spec.shape.as_list()}")

# 方法3：使用 batch
dataset3 = tf.data.Dataset.from_tensor_slices(string_tensor).batch(1)
print("\n方法3 - 使用 batch(1):")
print(f"  element_spec: {dataset3.element_spec}")
print(f"  shape: {dataset3.element_spec.shape}")
print(f"  shape.as_list(): {dataset3.element_spec.shape.as_list()}")

# 方法4：使用 batch(2)
dataset4 = tf.data.Dataset.from_tensor_slices(string_tensor).batch(2)
print("\n方法4 - 使用 batch(2):")
print(f"  element_spec: {dataset4.element_spec}")
print(f"  shape: {dataset4.element_spec.shape}")
print(f"  shape.as_list(): {dataset4.element_spec.shape.as_list()}")