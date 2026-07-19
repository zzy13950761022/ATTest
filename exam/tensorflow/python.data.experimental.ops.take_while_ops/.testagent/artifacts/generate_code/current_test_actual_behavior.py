import warnings
import tensorflow as tf
from tensorflow.python.data.experimental.ops.take_while_ops import take_while

# Test actual behavior of take_while with non-function arguments
print("Testing take_while with non-function arguments...")

test_cases = [
    (None, "None"),
    ("not_a_function", "string"),
    (123, "integer"),
    ([1, 2, 3], "list"),
    ({"key": "value"}, "dict"),
]

for value, description in test_cases:
    print(f"\nTesting with {description}: {value}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = take_while(value)
            print(f"  Result: {result}")
            print(f"  Type: {type(result)}")
            print(f"  Callable: {callable(result)}")
            
            # Try to use it with a dataset
            dataset = tf.data.Dataset.range(5)
            try:
                transformed = result(dataset)
                print(f"  Can be applied to dataset: Yes")
                print(f"  Transformed type: {type(transformed)}")
            except Exception as e:
                print(f"  Error when applying to dataset: {type(e).__name__}: {e}")
                
    except Exception as e:
        print(f"  Error in take_while call: {type(e).__name__}: {e}")