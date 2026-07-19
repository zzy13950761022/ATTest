import torch
import torch.nn as nn
from unittest.mock import Mock
import io

# 创建一个简单的模块
class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.linear(x))

# 编译为 ScriptModule
module = torch.jit.script(SimpleModule())

# 测试 torch.jit.save 是否会调用 flush
mock_file = Mock(spec=['write', 'flush'])
buffer_content = None

def mock_write_side_effect(data):
    nonlocal buffer_content
    buffer_content = data
    return len(data) if data else 0

mock_file.write.side_effect = mock_write_side_effect
mock_file.flush.return_value = None

# 调用 torch.jit.save
torch.jit.save(module, mock_file)

print(f"write called: {mock_file.write.called}")
print(f"flush called: {mock_file.flush.called}")
print(f"write call count: {mock_file.write.call_count}")
print(f"flush call count: {mock_file.flush.call_count}")

# 测试真实文件对象
real_buffer = io.BytesIO()
torch.jit.save(module, real_buffer)
print(f"\nReal buffer size: {len(real_buffer.getvalue())} bytes")

# 测试 torch.jit.load 的行为
real_buffer.seek(0)
loaded_module = torch.jit.load(real_buffer)
print(f"Module loaded successfully: {loaded_module is not None}")