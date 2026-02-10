# torch.ao.quantization.fuse_modules - 函数说明

## 1. 基本信息
- **FQN**: torch.ao.quantization.fuse_modules
- **模块文件**: `/opt/anaconda3/envs/testagent-experiment/lib/python3.10/site-packages/torch/ao/quantization/fuse_modules.py`
- **签名**: (model, modules_to_fuse, inplace=False, fuser_func=<function fuse_known_modules at 0x112232560>, fuse_custom_config_dict=None)
- **对象类型**: function

## 2. 功能概述
将模型中的多个模块融合为单个模块。支持特定序列的融合：conv-bn、conv-bn-relu、conv-relu、linear-relu、bn-relu。融合后，列表中的第一个模块被替换为融合模块，其余模块替换为恒等映射。

## 3. 参数说明
- model (PyTorch Module): 包含待融合模块的模型
- modules_to_fuse (list): 模块名称列表。可以是字符串列表（单个融合组）或列表的列表（多个融合组）
- inplace (bool/False): 是否原地修改模型。默认返回新模型
- fuser_func (callable/fuse_known_modules): 融合函数，接收模块列表并返回等长的融合模块列表
- fuse_custom_config_dict (dict/None): 自定义融合配置字典

## 4. 返回值
- 类型: PyTorch Module
- 结构: 包含融合模块的模型
- 特性: 当 inplace=False 时返回新副本

## 5. 文档要点
- 仅支持特定模块序列的融合
- 模型需处于 eval() 模式
- 融合后前向钩子会丢失（部分会转移到融合模块）
- 支持自定义融合方法映射

## 6. 源码摘要
- 关键路径: 检查 modules_to_fuse 类型 → 调用 _fuse_modules → 深度复制模型（非原地）→ 遍历融合组 → 调用 _fuse_modules_helper
- 依赖函数: _get_module, _set_module, fuse_known_modules, get_fuser_method
- 副作用: 修改模型结构，可能丢失钩子，非原地时创建模型副本

## 7. 示例与用法
```python
# 多个融合组
modules_to_fuse = [['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)

# 单个融合组
modules_to_fuse = ['conv1', 'bn1', 'relu1']
fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
```

## 8. 风险与空白
- 未明确指定支持的 Conv/Linear/BatchNorm 具体类型（Conv1d/2d/3d, Linear, BatchNorm1d/2d/3d）
- 未说明不支持的模块序列如何处理（文档说"left unchanged"但源码可能抛出异常）
- 自定义融合配置的具体格式和限制未详细说明
- 未明确说明融合对模型训练状态的影响
- 未提供融合后模块命名的具体规则