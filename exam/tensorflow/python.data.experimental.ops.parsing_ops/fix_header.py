"""Fix the create_string_dataset function"""
import tensorflow as tf

def create_string_dataset_fixed(serialized_examples, batch_size=1):
    """Create a dataset of serialized Example protos as string vectors.
    
    parse_example_dataset requires input to be a dataset of string vectors
    (shape=[None]), not scalar strings.
    
    Args:
        serialized_examples: List of serialized Example protos
        batch_size: Batch size for the dataset. Default is 1.
    
    Returns:
        A dataset where each element is a string vector of shape [batch_size]
    """
    # 创建标量字符串数据集
    dataset = tf.data.Dataset.from_tensor_slices(serialized_examples)
    # 应用批处理，将标量字符串转换为字符串向量
    dataset = dataset.batch(batch_size)
    return dataset

# 测试函数
def test_create_string_dataset():
    # 创建一些测试数据
    test_examples = [b"example1", b"example2", b"example3", b"example4"]
    
    print("测试 batch_size=1:")
    dataset1 = create_string_dataset_fixed(test_examples, batch_size=1)
    print(f"  element_spec: {dataset1.element_spec}")
    print(f"  形状: {dataset1.element_spec.shape}")
    
    print("\n测试 batch_size=2:")
    dataset2 = create_string_dataset_fixed(test_examples, batch_size=2)
    print(f"  element_spec: {dataset2.element_spec}")
    print(f"  形状: {dataset2.element_spec.shape}")
    
    print("\n迭代数据集 (batch_size=2):")
    for i, element in enumerate(dataset2):
        print(f"  元素 {i}: 形状={element.shape}, 值={element.numpy()}")

if __name__ == "__main__":
    test_create_string_dataset()