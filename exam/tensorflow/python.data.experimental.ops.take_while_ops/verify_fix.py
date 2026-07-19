import warnings
import tensorflow as tf
from tensorflow.python.data.experimental.ops.take_while_ops import take_while

print("Verifying CASE_06 fix...")

# Test CASE_06 scenarios
test_cases = [
    ("returns_integer", lambda x: tf.constant(42), ValueError),
    ("returns_string", lambda x: tf.constant("not_a_boolean"), ValueError),
]

all_passed = True
for desc, predicate, expected_error in test_cases:
    print(f"\nTesting {desc}:")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        
        try:
            # Step 1: take_while should succeed and return a callable
            transform_func = take_while(predicate)
            print(f"  ✓ take_while succeeded, returned callable")
            
            # Step 2: Try to apply to dataset - should raise ValueError
            dataset = tf.data.Dataset.range(5)
            try:
                transformed = transform_func(dataset)
                print(f"  ✗ transform_func succeeded (should have raised {expected_error.__name__})")
                all_passed = False
            except expected_error as e:
                print(f"  ✓ transform_func raised {type(e).__name__} as expected")
                error_msg = str(e)
                if "predicate" in error_msg.lower() and ("bool" in error_msg.lower() or "boolean" in error_msg.lower()):
                    print(f"  ✓ Error message contains relevant info: {error_msg[:100]}...")
                else:
                    print(f"  ⚠ Error message may not contain expected info: {error_msg[:100]}...")
            except Exception as e:
                print(f"  ✗ transform_func raised unexpected error: {type(e).__name__}: {e}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ take_while failed unexpectedly: {type(e).__name__}: {e}")
            all_passed = False

if all_passed:
    print("\n✅ All tests passed! CASE_06 fix is correct.")
else:
    print("\n❌ Some tests failed. CASE_06 fix needs revision.")