import tensorflow as tf
from tensorflow.python.feature_column.sequence_feature_column import sequence_categorical_column_with_identity

# Create a sequence categorical column
column = sequence_categorical_column_with_identity(
    key="test_feature", 
    num_buckets=10, 
    default_value=0
)

print("Column type:", type(column))
print("Column class name:", column.__class__.__name__)
print("\nAll attributes and methods:")
for attr in dir(column):
    if not attr.startswith('__'):
        print(f"  {attr}")

print("\nChecking for specific methods:")
print(f"  has _get_sequence_dense_tensor: {hasattr(column, '_get_sequence_dense_tensor')}")
print(f"  has _sequence_length: {hasattr(column, '_sequence_length')}")
print(f"  has get_sparse_tensors: {hasattr(column, 'get_sparse_tensors')}")
print(f"  has categorical_column: {hasattr(column, 'categorical_column')}")

if hasattr(column, 'categorical_column'):
    cat_col = column.categorical_column
    print(f"\nCategorical column type: {type(cat_col)}")
    print(f"Categorical column class name: {cat_col.__class__.__name__}")
    print(f"Categorical column key: {cat_col.key}")
    print(f"Categorical column num_buckets: {cat_col.num_buckets}")
    print(f"Categorical column default_value: {cat_col.default_value}")