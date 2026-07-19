import os

def check_file_size(filepath):
    """Check file size and line count."""
    try:
        # Get file size in bytes
        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024
        
        # Count lines
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            line_count = len(lines)
        
        print(f"File: {filepath}")
        print(f"Size: {size_bytes} bytes ({size_kb:.2f} KB)")
        print(f"Lines: {line_count}")
        
        # Check constraints
        if size_kb > 8:
            print("⚠️  WARNING: File size exceeds 8KB limit")
        else:
            print("✓ File size within 8KB limit")
        
        if line_count > 200:
            print("⚠️  WARNING: Line count exceeds 200 lines")
        else:
            print("✓ Line count within 200 lines")
        
        return size_kb <= 8, line_count <= 200
    except Exception as e:
        print(f"Error: {e}")
        return False, False

if __name__ == '__main__':
    test_file = 'tests/test_tensorflow_python_ops_functional_ops.py'
    size_ok, lines_ok = check_file_size(test_file)
    
    if size_ok and lines_ok:
        print("\n✓ All constraints satisfied")
    else:
        print("\n✗ Some constraints violated")