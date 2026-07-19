import numpy as np

# Check if bfloat16 is available in numpy
print("Checking numpy version:", np.__version__)
print("Available dtypes:", [dtype for dtype in dir(np) if 'float' in dtype.lower()])

# Try to create bfloat16 array
try:
    # Check if np.bfloat16 exists
    if hasattr(np, 'bfloat16'):
        print("np.bfloat16 exists")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.bfloat16)
        print("bfloat16 array created:", arr)
        print("dtype:", arr.dtype)
    else:
        print("np.bfloat16 does not exist")
        
        # Check for alternative bfloat16 support
        import tensorflow as tf
        print("\nChecking TensorFlow bfloat16 support:")
        print("tf.dtypes.bfloat16:", tf.dtypes.bfloat16)
        
        # Create bfloat16 tensor
        tf_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.bfloat16)
        print("TF bfloat16 tensor:", tf_tensor)
        print("TF tensor dtype:", tf_tensor.dtype)
        
        # Convert to numpy
        np_array = tf_tensor.numpy()
        print("Converted to numpy:", np_array)
        print("Numpy dtype:", np_array.dtype)
        
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()