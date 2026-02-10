# torch.utils.cpp_extension 测试计划

## 1. 测试策略
- 单元测试框架：pytest
- 隔离策略：mock/monkeypatch/fixtures 隔离外部依赖
- 随机性处理：无随机性要求，固定测试数据
- 构建隔离：使用临时目录，避免污染系统
- 平台兼容：通过mock模拟不同平台行为

## 2. 生成规格摘要（来自 test_plan.json）
- **SMOKE_SET**: CASE_01, CASE_02, CASE_03, CASE_04
- **DEFERRED_SET**: CASE_05, CASE_06, CASE_07, CASE_08
- **group列表**: 
  - G1: 核心编译与加载功能（CASE_01-02, CASE_05-06）
  - G2: 参数组合与错误处理（CASE_03-04, CASE_07-08）
- **active_group_order**: G1, G2
- **断言分级策略**: 首轮使用weak断言，最终轮启用strong断言
- **预算策略**: 
  - S类用例：max_lines=60-70, max_params=4-5
  - M类用例：max_lines=85-90, max_params=6-7
- **迭代策略**: 
  - round1: 仅SMOKE_SET，weak断言，最多5个用例
  - roundN: 修复失败用例，提升deferred用例
  - final: 启用strong断言，可选覆盖率

## 3. 数据与边界
- **正常数据集**: 简单C++源文件，混合C++/CUDA源文件
- **随机生成策略**: 固定测试源文件内容，不随机生成
- **边界值**: 
  - sources为单个字符串（非列表）
  - 空sources列表触发异常
  - 无效文件路径触发异常
  - build_directory为None使用临时目录
  - with_cuda=None自动检测CUDA
- **极端形状**: 复杂编译器标志组合，多个包含路径
- **空输入**: sources为空列表（异常场景）
- **负例与异常场景**:
  1. 无效源文件路径
  2. 缺少CUDA工具链但包含CUDA源文件
  3. 无效编译器标志导致编译失败
  4. 名称与pybind11模块名不匹配
  5. 磁盘空间不足（模拟）
  6. 权限问题（模拟）

## 4. 覆盖映射
- **TC-01 (CASE_01)**: 基本C++扩展编译加载 → 需求2.1, 4.1
- **TC-02 (CASE_02)**: 混合C++/CUDA扩展编译 → 需求2.2, 4.1
- **TC-03 (CASE_03)**: 参数组合测试 → 需求2.3, 4.2
- **TC-04 (CASE_04)**: 无效源文件错误处理 → 需求4.1, 4.3
- **TC-05 (CASE_05)**: 单字符串sources参数 → 需求4.2

- **尚未覆盖的风险点**:
  1. 平台特定行为差异（Windows/Linux/macOS）
  2. CUDA版本兼容性问题
  3. 编译器版本依赖
  4. 并发编译安全性
  5. 真实磁盘空间不足处理
  6. 环境变量影响（TORCH_EXTENSIONS_DIR）

## 5. Mock策略
- **核心mock目标**: subprocess.run（编译过程）
- **模块加载mock**: importlib.import_module
- **文件系统mock**: os.path, tempfile, shutil
- **CUDA检测mock**: torch.cuda.is_available
- **环境变量mock**: os.environ

## 6. 测试文件组织
- 主测试文件: `tests/test_torch_utils_cpp_extension.py`
- 分组文件: 
  - `tests/test_torch_utils_cpp_extension_g1.py` (G1组)
  - `tests/test_torch_utils_cpp_extension_g2.py` (G2组)
- 共享fixtures: conftest.py中定义mock配置和测试数据