import warnings
import tensorflow as tf
from tensorflow.python.data.experimental.ops.take_while_ops import take_while

print("Testing error timing for take_while with non-function arguments...")

test_cases = [
    (None, "None"),
    ("not_a_function", "string"),
    (123, "integer"),
]

for value, description in test_cases:
    print(f"\n=== Testing with {description}: {value} ===")
    
    # Step 1: Call take_while
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            print("1. Calling take_while...")
            transform_func = take_while(value)
            print(f"   Success! Got transform function: {transform_func}")
            print(f"   Type: {type(transform_func)}")
            print(f"   Callable: {callable(transform_func)}")
    except Exception as e:
        print(f"   Error in take_while call: {type(e).__name__}: {e}")
        continue
    
    # Step 2: Try to apply to dataset
    try:
        dataset = tf.data.Dataset.range(5)
        print("2. Applying transform to dataset...")
        transformed = transform_func(dataset)
        print(f"   Success! Got transformed dataset: {transformed}")
        print(f"   Type: {type(transformed)}")
    except Exception as e:
        print(f"   Error when applying to dataset: {type(e).__name__}: {e}")
        continue
    
    # Step 3: Try to iterate
    try:
        print("3. Iterating over transformed dataset...")
        result = list(transformed.as_numpy_iterator())
        print(f"   Success! Got result: {result}")
    except Exception as e:
        print(f"   Error during iteration: {type(e).__name__}: {e}")
        continue