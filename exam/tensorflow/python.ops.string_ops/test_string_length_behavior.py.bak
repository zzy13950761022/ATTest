import tensorflow as tf
from tensorflow.python.ops import string_ops

# Test 1: Special characters
print("Test 1: Special characters")
test_str = "a\nb\tc"
print(f"String: {repr(test_str)}")
print(f"Python len: {len(test_str)}")
print(f"Python bytes: {len(test_str.encode('utf-8'))}")

tensor = tf.constant([test_str])
result = string_ops.string_length(tensor, unit="BYTE")
print(f"TensorFlow BYTE length: {result.numpy()[0]}")

# Test 2: Complex emoji
print("\nTest 2: Complex emoji")
family_emoji = "👨‍👩‍👧‍👦"
print(f"Emoji: {family_emoji}")
print(f"Python len: {len(family_emoji)}")
print(f"Python bytes: {len(family_emoji.encode('utf-8'))}")

tensor2 = tf.constant([family_emoji, "a"])
result_byte = string_ops.string_length(tensor2, unit="BYTE")
result_char = string_ops.string_length(tensor2, unit="UTF8_CHAR")
print(f"TensorFlow BYTE length: {result_byte.numpy()}")
print(f"TensorFlow UTF8_CHAR length: {result_char.numpy()}")

# Test 3: Tab character
print("\nTest 3: Tab character")
tab_str = "\t"
print(f"Tab repr: {repr(tab_str)}")
print(f"Python len: {len(tab_str)}")
print(f"Python bytes: {len(tab_str.encode('utf-8'))}")

tensor3 = tf.constant([tab_str])
result3 = string_ops.string_length(tensor3, unit="BYTE")
print(f"TensorFlow BYTE length: {result3.numpy()[0]}")

# Test 4: Mixed newlines and tabs
print("\nTest 4: Mixed newlines and tabs")
mixed1 = "line1\nline2"
mixed2 = "col1\tcol2\tcol3"
print(f"mixed1: {repr(mixed1)}, Python bytes: {len(mixed1.encode('utf-8'))}")
print(f"mixed2: {repr(mixed2)}, Python bytes: {len(mixed2.encode('utf-8'))}")

tensor4 = tf.constant([mixed1, mixed2])
result4 = string_ops.string_length(tensor4, unit="BYTE")
print(f"TensorFlow BYTE lengths: {result4.numpy()}")