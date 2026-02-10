# tensorflow.python.saved_model.save - 函数说明

## 1. 基本信息
- **FQN**: tensorflow.python.saved_model.save
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/tensorflow/python/saved_model/save.py`
- **签名**: save(obj, export_dir, signatures=None, options=None)
- **对象类型**: function

## 2. 功能概述
将可追踪的 Python 对象（如 tf.Module）导出为 SavedModel 格式。该函数序列化对象及其依赖项，包括变量、函数和签名，保存到指定目录。主要用于模型持久化和部署。

## 3. 参数说明
- obj (Trackable/必需): 可追踪对象，必须继承自 Trackable 类（如 tf.Module、tf.train.Checkpoint）
- export_dir (str/必需): 保存 SavedModel 的目录路径
- signatures (多种类型/可选): 
  - 带输入签名的 tf.function（使用默认服务签名键）
  - f.get_concrete_function() 的结果
  - 字典：映射签名键到 tf.function 实例或具体函数
- options (SaveOptions/可选): 保存配置选项对象

## 4. 返回值
- 无返回值（None），但会创建文件系统结构
- 在指定目录生成 SavedModel 文件（saved_model.pb）和相关资源

## 5. 文档要点
- obj 必须继承自 Trackable 类
- 变量必须通过分配给跟踪对象的属性来跟踪
- 如果 signatures 参数省略，会自动搜索 @tf.function 装饰的方法
- 不支持在函数体内调用（不能在 @tf.function 内部使用）
- 在 TensorFlow 1.x 图形构建模式下支持不佳

## 6. 源码摘要
- 主要调用链：save → save_and_return_nodes → _build_meta_graph → _build_meta_graph_impl
- 关键组件：_AugmentedGraphView（扩展对象图）、_SaveableView（冻结视图）
- 依赖的辅助函数：signature_serialization、function_serialization、builder_impl
- 副作用：文件 I/O（创建目录、写入文件）、图形操作、资源序列化

## 7. 示例与用法
```python
class Adder(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def add(self, x):
        return x + x

model = Adder()
tf.saved_model.save(model, '/tmp/adder')
```

## 8. 风险与空白
- 目标 FQN 指向模块而非函数，模块包含多个函数（save 和 save_and_return_nodes）
- 需要测试：不同类型的 Trackable 对象、各种签名格式、选项配置
- 边界情况：空签名、无效导出目录、非跟踪对象、嵌套函数
- 缺少信息：具体错误类型和异常处理细节、性能影响
- 需要在测试中覆盖：资源变量处理、资产文件、自定义梯度选项