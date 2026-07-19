import tensorflow.python.compat.compat as compat

# 测试无效参数的行为
print("Testing invalid parameters...")

# 测试无效月份 (13)
try:
    result = compat.forward_compatible(2021, 13, 1)
    print(f"forward_compatible(2021, 13, 1) = {result} (no exception)")
except Exception as e:
    print(f"forward_compatible(2021, 13, 1) raised {type(e).__name__}: {e}")

# 测试无效日期 (2月30日)
try:
    result = compat.forward_compatible(2021, 2, 30)
    print(f"forward_compatible(2021, 2, 30) = {result} (no exception)")
except Exception as e:
    print(f"forward_compatible(2021, 2, 30) raised {type(e).__name__}: {e}")

# 测试月份边界值0
try:
    result = compat.forward_compatible(2021, 0, 1)
    print(f"forward_compatible(2021, 0, 1) = {result} (no exception)")
except Exception as e:
    print(f"forward_compatible(2021, 0, 1) raised {type(e).__name__}: {e}")

# 测试日期边界值0
try:
    result = compat.forward_compatible(2021, 12, 0)
    print(f"forward_compatible(2021, 12, 0) = {result} (no exception)")
except Exception as e:
    print(f"forward_compatible(2021, 12, 0) raised {type(e).__name__}: {e}")

# 测试无效年份类型
try:
    result = compat.forward_compatible("invalid", 12, 1)
    print(f"forward_compatible('invalid', 12, 1) = {result} (no exception)")
except Exception as e:
    print(f"forward_compatible('invalid', 12, 1) raised {type(e).__name__}: {e}")