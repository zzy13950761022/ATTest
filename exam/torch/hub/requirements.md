# torch.hub 测试需求

## 1. 目标与范围
- 主要功能与期望行为
  - 从 GitHub 仓库或本地目录加载模型/可调用对象
  - 支持远程下载和本地加载两种模式
  - 通过 hubconf.py 定义的入口点调用目标函数
  - 管理缓存和信任机制
- 不在范围内的内容
  - torch.hub 模块的其他函数（list, help, download_url_to_file 等）
  - 模型训练或推理的具体实现
  - 第三方仓库的内容验证

## 2. 输入与约束
- 参数列表（名称、类型/shape、默认值）
  - repo_or_dir: str, GitHub格式 "owner/repo[:ref]" 或本地路径
  - model: str, hubconf.py 中定义的入口点名称
  - *args: 任意, 传递给 model 的位置参数
  - source: str, 'github' 或 'local', 默认 'github'
  - trust_repo: bool/str/None, 默认 None
  - force_reload: bool, 默认 False
  - verbose: bool, 默认 True
  - skip_validation: bool, 默认 False
  - **kwargs: 任意, 传递给 model 的关键字参数
- 有效取值范围/维度/设备要求
  - GitHub 仓库格式必须符合 "owner/repo[:tag_or_branch]" 模式
  - 本地路径必须存在且包含 hubconf.py 文件
  - source 只能为 'github' 或 'local'
  - trust_repo 接受 False/True/"check"/None
- 必需与可选组合
  - repo_or_dir 和 model 为必需参数
  - 其他参数均为可选，有默认值
- 随机性/全局状态要求
  - 依赖全局缓存目录（~/.cache/torch/hub）
  - 信任列表存储在缓存中
  - 网络状态影响 GitHub 源加载

## 3. 输出与判定
- 期望返回结构及关键字段
  - 返回 model 可调用对象的调用结果
  - 通常为 torch.nn.Module 实例，但类型由具体模型决定
  - 必须包含模型定义的所有必要属性和方法
- 容差/误差界（如浮点）
  - 不适用（返回对象而非数值）
- 状态变化或副作用检查点
  - 缓存目录应有新文件（GitHub 源）
  - 信任列表可能更新（trust_repo=True）
  - 控制台输出（verbose=True）
  - 网络请求记录（GitHub API 调用）

## 4. 错误与异常场景
- 非法输入/维度/类型触发的异常或警告
  - 无效 GitHub 仓库格式（非 "owner/repo[:ref]"）
  - 不存在的本地路径
  - 缺失 hubconf.py 文件
  - 不存在的入口点名称
  - 错误的 source 参数值
  - 无效的 trust_repo 值
- 边界值（空、None、0 长度、极端形状/数值）
  - repo_or_dir 为空字符串
  - model 为空字符串
  - ref 部分为空（"owner/repo:"）
  - 超长仓库名称
  - 特殊字符路径

## 5. 依赖与环境
- 外部资源/设备/网络/文件依赖
  - GitHub API 访问（网络连接）
  - 本地文件系统权限
  - GITHUB_TOKEN 环境变量（认证）
  - Python importlib 模块
  - urllib.request 网络库
- 需要 mock/monkeypatch 的部分
  - 网络请求（GitHub 下载）
  - 文件系统操作（缓存读写）
  - 用户输入（信任提示）
  - 环境变量（GITHUB_TOKEN）
  - 时间相关操作（缓存过期）

## 6. 覆盖与优先级
- 必测路径（高优先级，最多 5 条，短句）
  1. GitHub 仓库正常加载流程（标准格式）
  2. 本地路径加载流程（有效 hubconf.py）
  3. trust_repo 不同值的信任处理
  4. force_reload 强制重新下载行为
  5. 参数传递给模型入口点
- 可选路径（中/低优先级合并为一组列表）
  - 不同 ref 格式（tag/branch/commit）
  - verbose=False 静默模式
  - skip_validation=True 跳过验证
  - 缓存命中/未命中场景
  - 网络超时/失败处理
  - 权限不足的本地路径
  - 并发加载同一仓库
- 已知风险/缺失信息（仅列条目，不展开）
  - 返回值类型未在签名中明确
  - 信任机制的详细交互行为
  - 缓存清理和过期策略
  - 网络重试和错误恢复机制
  - 内存使用和资源释放