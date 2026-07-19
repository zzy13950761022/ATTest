#!/usr/bin/env python3
"""
验证对测试文件的修复
"""

import ast
import sys

def check_file_for_backend_issues(filepath):
    """检查文件中是否有backend属性相关的问题"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查是否有 pg.backend.return_value 或 pg.backend = 这样的模式
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'pg.backend' in line and ('return_value' in line or '=' in line):
                issues.append(f"第{i}行: {line.strip()}")
                
    except Exception as e:
        issues.append(f"读取文件时出错: {e}")
    
    return issues

def main():
    files_to_check = [
        'tests/test_torch_distributed_distributed_c10d_g1.py',
        'tests/test_torch_distributed_distributed_c10d_g2.py',
        'tests/test_torch_distributed_distributed_c10d_g3.py',
        'tests/test_torch_distributed_distributed_c10d_g3_fixed.py'
    ]
    
    all_issues = []
    
    print("检查测试文件中的backend属性问题...")
    print("=" * 60)
    
    for filepath in files_to_check:
        print(f"\n检查文件: {filepath}")
        issues = check_file_for_backend_issues(filepath)
        
        if issues:
            print(f"  发现 {len(issues)} 个问题:")
            for issue in issues:
                print(f"  - {issue}")
            all_issues.extend([(filepath, issue) for issue in issues])
        else:
            print("  未发现问题")
    
    print("\n" + "=" * 60)
    if all_issues:
        print(f"总共发现 {len(all_issues)} 个问题:")
        for filepath, issue in all_issues:
            print(f"  {filepath}: {issue}")
        return 1
    else:
        print("所有文件检查通过！")
        return 0

if __name__ == '__main__':
    sys.exit(main())