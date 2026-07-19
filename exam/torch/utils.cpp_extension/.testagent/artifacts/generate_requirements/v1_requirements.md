# torch.utils.cpp_extension 测试需求

## 1. 目标与范围
- 主要功能与期望行为：验证 C++/CUDA 扩展的即时编译和加载功能，确保正确编译源文件并返回可用模块
- 不在范围内的内容：不测试其他扩展函数（load_inline、CppExtension 等），不验证扩展模块内部逻辑

## 2. 输入与约束
- 参数列表：
  - name: str，必须与 pybind11 模块名匹配
  - sources: Union[str, List[str]]，C++/CUDA 源文件路径
  - extra_cflags: List[str]/None，C++ 编译器标志
  - extra_cuda_cflags: List[str]/None，nvcc 编译器标志
  - extra_ldflags: List[str]/None，链接器标志
  - extra_include_paths: List[str]/None，包含目录
  - build_directory: str/None，构建目录路径
  - verbose: bool，默认 False
  - with_cuda: Optional[bool]，None 时自动检测
  - is_python_module: bool，默认 True
  - is_standalone: bool，默认 False
  - keep_intermediates: bool，默认 True

- 有效取值范围/维度/设备要求：
  - sources 必须包含至少一个有效源文件
  - 源文件扩展名需为 .cpp/.cc/.cxx/.cu/.cuh
  - 需要兼容的 C++ 编译器（gcc/clang/MSVC）
  - CUDA 源文件需要 CUDA 工具链支持

- 必需与可选组合：
  - name 和 sources 为必需参数
  - 其他参数均为可选，有默认值

- 随机性/全局状态要求：
  - 无随机性要求
  - 可能修改环境变量和文件系统状态

## 3. 输出与判定
- 期望返回结构及关键字段：
  - is_python_module=True：返回 Python 模块对象，可调用其中函数
  - is_python_module=False, is_standalone=False：返回 None
  - is_standalone=True：返回可执行文件路径字符串

- 容差/误差界：
  - 无浮点误差要求
  - 编译时间可能因系统而异

- 状态变化或副作用检查点：
  - 构建目录创建和文件写入
  - 进程创建（编译命令执行）
  - 动态库加载到 Python 进程

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常：
  - sources 为空列表或无效路径
  - name 与 pybind11 模块名不匹配
  - 无效编译器标志导致编译失败
  - 缺少 CUDA 工具链但包含 CUDA 源文件

- 边界值：
  - sources 为单个字符串而非列表
  - build_directory 为 None（使用临时目录）
  - with_cuda=None（自动检测）
  - 极端形状/数值：超大源文件、复杂编译标志

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖：
  - Ninja 构建系统
  - C++ 编译器（gcc/clang/MSVC）
  - CUDA 工具链（可选，用于 CUDA 编译）
  - 磁盘空间用于构建目录

- 需要 mock/monkeypatch 的部分：
  - `subprocess.run`：模拟编译过程
  - `importlib.import_module`：模拟模块加载
  - `tempfile.mkdtemp`：控制临时目录创建
  - `os.path` 相关函数：文件路径操作
  - `torch.cuda.is_available`：控制 CUDA 可用性检测

## 6. 覆盖与优先级
- 必测路径（高优先级）：
  1. 基本 C++ 扩展编译和加载
  2. 混合 C++/CUDA 扩展编译（CUDA 可用时）
  3. 参数组合：verbose=True, extra_cflags 设置
  4. 错误处理：无效源文件路径
  5. 边界情况：sources 为单个字符串

- 可选路径（中/低优先级）：
  - is_standalone=True 模式
  - keep_intermediates=False 行为
  - 复杂编译器标志组合
  - 自定义 build_directory
  - 环境变量影响（TORCH_EXTENSIONS_DIR）

- 已知风险/缺失信息：
  - 平台特定行为差异（Windows/Linux/macOS）
  - CUDA 版本兼容性问题
  - 编译器版本依赖
  - 磁盘空间不足处理
  - 并发编译安全性