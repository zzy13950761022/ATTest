"""
测试 tensorflow.python.data.experimental.ops.parsing_ops 模块
G2组：特征类型与边界测试
"""
import math
import pytest
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parsing_ops
from unittest import mock

# ==== BLOCK:HEADER START ====
# 测试类定义和公共fixture
class TestParsingOpsG2:
    """G2组测试类：特征类型与边界测试"""
    
    @pytest.fixture
    def tf_random_seed(self):
        """设置TensorFlow随机种子"""
        tf.random.set_seed(42)
        return 42
    
    @pytest.fixture
    def create_string_vector_dataset(self, tf_random_seed):
        """创建字符串向量数据集的辅助函数
        
        返回形状为[None]的字符串向量数据集，符合parse_example_dataset的输入要求。
        """
        def _create_dataset(size, feature_specs):
            """根据特征规格创建测试数据集
            
            Args:
                size: 数据集大小
                feature_specs: 特征规格字典
                
            Returns:
                包含序列化Example protos的字符串向量数据集（shape=[None]）
            """
            # 创建示例数据
            examples = []
            for i in range(size):
                # 创建序列化Example
                example = tf.train.Example()
                for key, spec in feature_specs.items():
                    if isinstance(spec, tf.io.FixedLenFeature):
                        if spec.dtype == tf.float32:
                            example.features.feature[key].float_list.value.extend([float(i)])
                        elif spec.dtype == tf.int64:
                            example.features.feature[key].int64_list.value.extend([i])
                        elif spec.dtype == tf.string:
                            example.features.feature[key].bytes_list.value.extend([f"value_{i}".encode()])
                    elif isinstance(spec, tf.io.VarLenFeature):
                        if spec.dtype == tf.float32:
                            example.features.feature[key].float_list.value.extend([float(i), float(i+1)])
                        elif spec.dtype == tf.int64:
                            example.features.feature[key].int64_list.value.extend([i, i+1])
                
                examples.append(example.SerializeToString())
            
            # 创建字符串向量数据集（shape=[None]）
            # 创建包含字符串向量的数据集
            # 每个元素是一个包含单个字符串的向量（shape=[1]）
            dataset = tf.data.Dataset.from_tensor_slices(examples)
            # 使用batch(1)将标量字符串转换为字符串向量
            dataset = dataset.batch(1)
            return dataset
        
        return _create_dataset
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_05 START ====
    @pytest.mark.parametrize("feature_types,dtype_combinations,deterministic,dataset_size", [
        (["FixedLenFeature", "VarLenFeature"], ["float32", "int64"], False, 15)
    ])
    def test_multiple_feature_types_support(self, create_string_vector_dataset, 
                                           feature_types, dtype_combinations, 
                                           deterministic, dataset_size):
        """测试多种特征类型支持 (TC-05)
        
        验证parse_example_dataset能够正确处理FixedLenFeature和VarLenFeature的组合。
        """
        # 创建特征规格字典
        features = {}
        
        # 添加FixedLenFeature
        if "FixedLenFeature" in feature_types:
            if "float32" in dtype_combinations:
                features["fixed_float"] = tf.io.FixedLenFeature([], tf.float32)
            if "int64" in dtype_combinations:
                features["fixed_int"] = tf.io.FixedLenFeature([], tf.int64)
        
        # 添加VarLenFeature
        if "VarLenFeature" in feature_types:
            if "float32" in dtype_combinations:
                features["var_float"] = tf.io.VarLenFeature(tf.float32)
            if "int64" in dtype_combinations:
                features["var_int"] = tf.io.VarLenFeature(tf.int64)
        
        # 创建测试数据集
        dataset = create_string_vector_dataset(dataset_size, features)
        
        # 验证输入数据集是字符串向量
        element_spec = dataset.element_spec
        assert isinstance(element_spec, tf.TensorSpec), "输入数据集应该是TensorSpec"
        assert element_spec.dtype == tf.string, "输入数据集应该是字符串类型"
        # parse_example_dataset期望字符串向量数据集（shape=[None]）
        # 使用batch(1)后，每个元素是形状为[1]的字符串向量
        assert element_spec.shape.as_list() == [None], "输入数据集应该是字符串向量(shape=[None])"
        
        # 应用parse_example_dataset转换
        parse_fn = parsing_ops.parse_example_dataset(
            features=features,
            num_parallel_calls=2,
            deterministic=deterministic
        )
        
        # 验证转换返回一个函数
        assert callable(parse_fn), "parse_example_dataset应该返回一个可调用函数"
        
        # 应用转换
        parsed_dataset = dataset.apply(parse_fn)
        
        # 验证输出数据集结构
        element_spec = parsed_dataset.element_spec
        assert isinstance(element_spec, dict), "输出element_spec应该是一个字典"
        
        # 检查所有特征都存在
        expected_keys = set(features.keys())
        assert set(element_spec.keys()) == expected_keys, \
            f"期望特征键: {expected_keys}, 实际: {set(element_spec.keys())}"
        
        # 验证每种特征类型的张量规格
        for key, feature in features.items():
            spec = element_spec[key]
            
            if isinstance(feature, tf.io.FixedLenFeature):
                # FixedLenFeature应该产生TensorSpec
                assert isinstance(spec, tf.TensorSpec), \
                    f"特征'{key}'应该是TensorSpec，实际是{type(spec)}"
                assert spec.dtype == feature.dtype, \
                    f"特征'{key}'的数据类型不匹配: 期望{feature.dtype}, 实际{spec.dtype}"
                # 注意：由于输入数据集是标量字符串，parse_example_dataset会内部处理batch
                # 注意：由于输入数据集是字符串向量(shape=[None])，输出形状会包含batch维度
                # 输出形状应该是[None] + 特征形状
                expected_shape = [None] + list(feature.shape)
                assert spec.shape.as_list() == expected_shape, \
                    f"特征'{key}'的形状不匹配: 期望{expected_shape}, 实际{spec.shape}"
            
            elif isinstance(feature, tf.io.VarLenFeature):
                # VarLenFeature应该产生SparseTensorSpec
                assert isinstance(spec, tf.SparseTensorSpec), \
                    f"特征'{key}'应该是SparseTensorSpec，实际是{type(spec)}"
                assert spec.dtype == feature.dtype, \
                    f"特征'{key}'的数据类型不匹配: 期望{feature.dtype}, 实际{spec.dtype}"
        
        # 验证数据集大小
        count = 0
        for element in parsed_dataset:
            count += 1
            
            # 验证所有键都存在
            assert set(element.keys()) == expected_keys, \
                f"元素{count}的特征键不匹配"
            
            # 验证每个特征的类型和形状
            for key, feature in features.items():
                tensor = element[key]
                
                if isinstance(feature, tf.io.FixedLenFeature):
                    # 验证FixedLenFeature
                    assert isinstance(tensor, tf.Tensor), \
                        f"特征'{key}'应该是Tensor，实际是{type(tensor)}"
                    assert tensor.dtype == feature.dtype, \
                        f"特征'{key}'的数据类型不匹配"
                    # 对于标量特征，输出应该包含batch维度
                    # 因为输入是字符串向量(shape=[1])，输出形状是[1] + 特征形状
                    expected_shape = tuple([1] + list(feature.shape))
                    assert tensor.shape == expected_shape, \
                        f"特征'{key}'的形状不匹配: 期望{expected_shape}, 实际{tensor.shape}"
                
                elif isinstance(feature, tf.io.VarLenFeature):
                    # 验证VarLenFeature
                    assert isinstance(tensor, tf.SparseTensor), \
                        f"特征'{key}'应该是SparseTensor，实际是{type(tensor)}"
                    assert tensor.dtype == feature.dtype, \
                        f"特征'{key}'的数据类型不匹配"
        
        assert count == dataset_size, f"期望{dataset_size}个元素，实际{count}个"
        
        # 验证特征数量
        expected_feature_count = 0
        if "FixedLenFeature" in feature_types:
            expected_feature_count += len(dtype_combinations)
        if "VarLenFeature" in feature_types:
            expected_feature_count += len(dtype_combinations)
        assert len(features) == expected_feature_count, \
            f"特征数量不匹配: 期望{expected_feature_count}, 实际{len(features)}"
        
        # 使用tf.io.parse_example作为参考实现进行比较
        # 注意：这里我们只比较FixedLenFeature，因为VarLenFeature的稀疏表示可能不同
        for i, element in enumerate(parsed_dataset):
            # 获取原始序列化数据
            original_example = list(dataset.take(1).as_numpy_iterator())[0]
            
            # 使用参考实现解析
            oracle_result = tf.io.parse_example(
                tf.constant([original_example]), 
                features
            )
            
            # 比较FixedLenFeature
            for key, feature in features.items():
                if isinstance(feature, tf.io.FixedLenFeature):
                    # element[key]是形状为[1]的张量，需要取第一个元素
                    actual_value = element[key][0]
                    oracle_value = oracle_result[key][0]
                    
                    # 根据数据类型进行比较
                    if feature.dtype == tf.float32:
                        tf.debugging.assert_near(
                            actual_value, oracle_value, 
                            rtol=1e-6, atol=1e-6,
                            message=f"特征'{key}'的值不匹配"
                        )
                    elif feature.dtype == tf.int64:
                        tf.debugging.assert_equal(
                            actual_value, oracle_value,
                            message=f"特征'{key}'的值不匹配"
                        )
            
            # 对于VarLenFeature，比较稀疏值
            for key, feature in features.items():
                if isinstance(feature, tf.io.VarLenFeature):
                    actual_sparse = element[key]
                    oracle_sparse = oracle_result[key][0]
                    
                    # 比较稀疏值
                    tf.debugging.assert_equal(
                        actual_sparse.values, oracle_sparse.values,
                        message=f"特征'{key}'的稀疏值不匹配"
                    )
                    
                    # 比较稀疏索引
                    # 注意：actual_sparse.indices包含batch维度，需要调整
                    # oracle_sparse.indices的形状是[n, 1]，actual_sparse.indices的形状是[n, 2]
                    # 其中第一列是batch索引（应该都是0）
                    if actual_sparse.indices.shape[0] > 0:
                        # 检查batch索引都是0
                        tf.debugging.assert_equal(
                            actual_sparse.indices[:, 0], 
                            tf.zeros([actual_sparse.indices.shape[0]], dtype=tf.int64),
                            message=f"特征'{key}'的batch索引不匹配"
                        )
                        # 比较特征索引（第二列）
                        tf.debugging.assert_equal(
                            actual_sparse.indices[:, 1:], 
                            oracle_sparse.indices,
                            message=f"特征'{key}'的稀疏索引不匹配"
                        )
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# 占位：deterministic参数行为
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# 占位：空数据集处理
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# 占位：无效输入数据集验证
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# 测试运行入口
if __name__ == "__main__":
    # 简单测试运行器用于调试
    import sys
    
    print("运行G2组测试...")
    
    # 导入测试类
    test_class = TestParsingOpsG2()
    
    # 运行CASE_05测试
    try:
        print("测试CASE_05: 多种特征类型支持...")
        # 创建fixture实例
        tf_random_seed = test_class.tf_random_seed()
        create_string_vector_dataset = test_class.create_string_vector_dataset(tf_random_seed)
        
        # 运行测试
        test_class.test_multiple_feature_types_support(
            create_string_vector_dataset,
            ["FixedLenFeature", "VarLenFeature"],
            ["float32", "int64"],
            False,
            15
        )
        print("✓ CASE_05: 多种特征类型支持 - 通过")
    except Exception as e:
        print(f"✗ CASE_05: 多种特征类型支持 - 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n所有测试通过!")
    
    # 也可以使用pytest运行
    # pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====