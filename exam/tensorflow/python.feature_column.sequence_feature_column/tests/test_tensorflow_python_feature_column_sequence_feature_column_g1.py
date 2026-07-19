"""
Test cases for tensorflow.python.feature_column.sequence_feature_column
Group G1: 核心序列特征列创建函数
"""

import os
import tempfile
import numpy as np
import pytest
import tensorflow as tf
from unittest import mock

# Import target functions
from tensorflow.python.feature_column.sequence_feature_column import (
    sequence_categorical_column_with_identity,
    sequence_categorical_column_with_hash_bucket,
    sequence_categorical_column_with_vocabulary_file,
    sequence_categorical_column_with_vocabulary_list,
    sequence_numeric_column,
    concatenate_context_input
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==== BLOCK:HEADER START ====
# Test class and fixtures for G1 tests
class TestSequenceFeatureColumnG1:
    """Test cases for core sequence feature column creation functions."""
    
    @pytest.fixture
    def temp_vocab_file(self):
        """Create a temporary vocabulary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for i in range(100):
                f.write(f"word_{i}\n")
            f.flush()
            yield f.name
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
    @pytest.mark.parametrize("func_name,params,expected_type,expect_failure", [
        ("sequence_categorical_column_with_identity", 
         {"key": "test_feature", "num_buckets": 10, "default_value": 0},
         "SequenceCategoricalColumn", False),
        ("sequence_categorical_column_with_hash_bucket",
         {"key": "test_feature", "hash_bucket_size": 100, "dtype": tf.int64},
         "SequenceCategoricalColumn", False),
        # Parameter extensions from test_plan.json
        # Note: default_value=-1 is invalid for num_buckets=1000, so we change it to 0
        ("sequence_categorical_column_with_identity",
         {"key": "extended_feature", "num_buckets": 1000, "default_value": 0},
         "SequenceCategoricalColumn", False),
        ("sequence_categorical_column_with_hash_bucket",
         {"key": "large_hash", "hash_bucket_size": 10000, "dtype": tf.string},
         "SequenceCategoricalColumn", False),
    ])
    def test_sequence_categorical_column_creation(self, func_name, params, expected_type, expect_failure):
        """Test basic creation of sequence categorical columns (CASE_01)."""
        # Get the function
        func = globals()[func_name]
        
        # Handle expected failures
        if expect_failure:
            with pytest.raises(ValueError):
                func(**params)
            return
        
        # Create the feature column
        column = func(**params)
        
        # Weak assertions
        # 1. Check instance type
        assert column is not None
        assert hasattr(column, '__class__')
        assert expected_type in column.__class__.__name__
        
        # 2. Check has key property - SequenceCategoricalColumn wraps categorical_column
        # The key is accessed through the categorical_column property
        assert hasattr(column, 'categorical_column')
        categorical_column = column.categorical_column
        assert hasattr(categorical_column, 'key')
        assert categorical_column.key == params['key']
        
        # 3. Check has dtype property
        if 'dtype' in params:
            assert hasattr(categorical_column, 'dtype')
            # Convert tf.dtype to string for comparison
            if hasattr(categorical_column.dtype, 'name'):
                assert categorical_column.dtype.name == params['dtype'].name
            else:
                assert str(categorical_column.dtype) == str(params['dtype'])
        
        # 4. Check has num_buckets or hash_bucket_size property
        if 'num_buckets' in params:
            assert hasattr(categorical_column, 'num_buckets')
            assert categorical_column.num_buckets == params['num_buckets']
        elif 'hash_bucket_size' in params:
            assert hasattr(categorical_column, 'hash_bucket_size')
            assert categorical_column.hash_bucket_size == params['hash_bucket_size']
        
        # 5. Check default_value if provided
        if 'default_value' in params:
            assert hasattr(categorical_column, 'default_value')
            assert categorical_column.default_value == params['default_value']
        
        # Additional basic checks
        assert hasattr(column, 'name')
        assert isinstance(column.name, str)
        
        # Check that it's a sequence column
        assert hasattr(column, '_is_v2_column')
        # Note: In TensorFlow 2.x, all feature columns are v2 columns
        
        # Check that the column has the expected properties for sequence processing
        # SequenceCategoricalColumn inherits from CategoricalColumn and has get_sparse_tensors method
        assert hasattr(column, 'get_sparse_tensors')
        
        # Check parse_example_spec for feature parsing
        assert hasattr(column, 'parse_example_spec')
        
        # Check num_buckets property
        assert hasattr(column, 'num_buckets')
        if 'num_buckets' in params:
            assert column.num_buckets == params['num_buckets']
        elif 'hash_bucket_size' in params:
            # For hash bucket columns, num_buckets should equal hash_bucket_size
            assert column.num_buckets == params['hash_bucket_size']
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
    def test_sequence_numeric_column_creation(self):
        """Test creation of sequence numeric column (CASE_02)."""
        # Test basic numeric column
        params = {
            "key": "numeric_feature",
            "shape": (10,),
            "dtype": tf.float32,
            "default_value": 0.0
        }
        
        column = sequence_numeric_column(**params)
        
        # Weak assertions
        # 1. Check instance type
        assert column is not None
        assert hasattr(column, '__class__')
        assert "SequenceNumericColumn" in column.__class__.__name__
        
        # 2. Check has key property
        assert hasattr(column, 'key')
        assert column.key == params['key']
        
        # 3. Check has dtype property
        assert hasattr(column, 'dtype')
        # Convert tf.dtype to string for comparison
        if hasattr(column.dtype, 'name'):
            assert column.dtype.name == params['dtype'].name
        else:
            assert str(column.dtype) == str(params['dtype'])
        
        # 4. Check has shape property
        assert hasattr(column, 'shape')
        # Convert shape to tuple for comparison
        column_shape = tuple(column.shape) if hasattr(column.shape, '__iter__') else column.shape
        expected_shape = tuple(params['shape']) if hasattr(params['shape'], '__iter__') else params['shape']
        assert column_shape == expected_shape
        
        # 5. Check default_value if provided
        assert hasattr(column, 'default_value')
        assert column.default_value == params['default_value']
        
        # Additional basic checks
        assert hasattr(column, 'name')
        assert isinstance(column.name, str)
        
        # Test with parameter extension (multi-dimensional shape)
        params_ext = {
            "key": "multi_dim_numeric",
            "shape": (5, 3),
            "dtype": tf.float32,
            "default_value": 1.0
        }
        
        column_ext = sequence_numeric_column(**params_ext)
        
        # Check shape for extended parameter
        assert hasattr(column_ext, 'shape')
        column_ext_shape = tuple(column_ext.shape) if hasattr(column_ext.shape, '__iter__') else column_ext.shape
        expected_ext_shape = tuple(params_ext['shape']) if hasattr(params_ext['shape'], '__iter__') else params_ext['shape']
        assert column_ext_shape == expected_ext_shape
        
        # Check default_value for extended parameter
        assert hasattr(column_ext, 'default_value')
        assert column_ext.default_value == params_ext['default_value']
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
    def test_vocabulary_file_sequence_column_creation(self, temp_vocab_file):
        """Test creation of sequence categorical column with vocabulary file (CASE_03)."""
        # Test basic vocabulary file column
        params = {
            "key": "vocab_feature",
            "vocabulary_file": temp_vocab_file,
            "vocabulary_size": 100
        }
        
        column = sequence_categorical_column_with_vocabulary_file(**params)
        
        # Weak assertions
        # 1. Check instance type
        assert column is not None
        assert hasattr(column, '__class__')
        assert "SequenceCategoricalColumn" in column.__class__.__name__
        
        # 2. Check has key property
        assert hasattr(column, 'categorical_column')
        categorical_column = column.categorical_column
        assert hasattr(categorical_column, 'key')
        assert categorical_column.key == params['key']
        
        # 3. Check has vocabulary_file property
        assert hasattr(categorical_column, 'vocabulary_file')
        # The vocabulary_file might be normalized, so we check it's not None
        assert categorical_column.vocabulary_file is not None
        
        # 4. Check has vocabulary_size property
        assert hasattr(categorical_column, 'vocabulary_size')
        assert categorical_column.vocabulary_size == params['vocabulary_size']
        
        # Additional basic checks
        assert hasattr(column, 'name')
        assert isinstance(column.name, str)
        
        # Check that it's a sequence column
        assert hasattr(column, '_is_v2_column')
        
        # Check sequence column properties
        assert hasattr(column, 'get_sparse_tensors')
        assert hasattr(column, 'parse_example_spec')
        assert hasattr(column, 'num_buckets')
        
        # Test with mock to ensure file access is handled properly
        with mock.patch('tensorflow.python.feature_column.sequence_feature_column.fc.categorical_column_with_vocabulary_file') as mock_cat_col:
            mock_cat_col.return_value = mock.MagicMock(
                key=params['key'],
                vocabulary_file=params['vocabulary_file'],
                vocabulary_size=params['vocabulary_size']
            )
            
            # Call the function with mocked dependency
            column_mock = sequence_categorical_column_with_vocabulary_file(**params)
            
            # Verify the mock was called with correct parameters
            mock_cat_col.assert_called_once_with(
                key=params['key'],
                vocabulary_file=params['vocabulary_file'],
                vocabulary_size=params['vocabulary_size'],
                dtype=tf.string,
                default_value=None,
                num_oov_buckets=0
            )
            
            # Verify the sequence column wraps the categorical column
            assert column_mock.categorical_column == mock_cat_col.return_value
        
        # Test error case: missing vocabulary_size
        with pytest.raises(ValueError) as exc_info:
            sequence_categorical_column_with_vocabulary_file(
                key="test_feature",
                vocabulary_file=temp_vocab_file
                # Missing vocabulary_size
            )
        
        # Check error message mentions vocabulary_size
        error_msg = str(exc_info.value).lower()
        assert 'vocabulary_size' in error_msg or 'required' in error_msg
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
    def test_vocabulary_list_sequence_column_creation(self):
        """Test creation of sequence categorical column with vocabulary list (CASE_04)."""
        # Test basic vocabulary list column
        params = {
            "key": "list_feature",
            "vocabulary_list": ["cat", "dog", "bird"],
            "dtype": tf.string
        }
        
        column = sequence_categorical_column_with_vocabulary_list(**params)
        
        # Weak assertions
        # 1. Check instance type
        assert column is not None
        assert hasattr(column, '__class__')
        assert "SequenceCategoricalColumn" in column.__class__.__name__
        
        # 2. Check has key property
        assert hasattr(column, 'categorical_column')
        categorical_column = column.categorical_column
        assert hasattr(categorical_column, 'key')
        assert categorical_column.key == params['key']
        
        # 3. Check has vocabulary_list property
        assert hasattr(categorical_column, 'vocabulary_list')
        # The vocabulary_list might be converted to tuple or list, so we check content
        vocab_list = categorical_column.vocabulary_list
        assert isinstance(vocab_list, (list, tuple))
        assert len(vocab_list) == len(params['vocabulary_list'])
        for expected_word in params['vocabulary_list']:
            assert expected_word in vocab_list
        
        # 4. Check has dtype property
        assert hasattr(categorical_column, 'dtype')
        # Convert tf.dtype to string for comparison
        if hasattr(categorical_column.dtype, 'name'):
            assert categorical_column.dtype.name == params['dtype'].name
        else:
            assert str(categorical_column.dtype) == str(params['dtype'])
        
        # Additional basic checks
        assert hasattr(column, 'name')
        assert isinstance(column.name, str)
        
        # Check that it's a sequence column
        assert hasattr(column, '_is_v2_column')
        
        # Check sequence column properties
        assert hasattr(column, 'get_sparse_tensors')
        assert hasattr(column, 'parse_example_spec')
        assert hasattr(column, 'num_buckets')
        
        # Test with different dtypes
        # Test with int64 dtype
        params_int = {
            "key": "int_feature",
            "vocabulary_list": [1, 2, 3, 4, 5],
            "dtype": tf.int64
        }
        
        column_int = sequence_categorical_column_with_vocabulary_list(**params_int)
        assert column_int is not None
        assert "SequenceCategoricalColumn" in column_int.__class__.__name__
        
        # Check vocabulary list for ints
        cat_col_int = column_int.categorical_column
        vocab_list_int = cat_col_int.vocabulary_list
        assert isinstance(vocab_list_int, (list, tuple))
        assert len(vocab_list_int) == len(params_int['vocabulary_list'])
        for expected_num in params_int['vocabulary_list']:
            assert expected_num in vocab_list_int
        
        # Test with default parameters (no dtype specified, should default to string)
        params_default = {
            "key": "default_feature",
            "vocabulary_list": ["apple", "banana", "cherry"]
        }
        
        column_default = sequence_categorical_column_with_vocabulary_list(**params_default)
        assert column_default is not None
        cat_col_default = column_default.categorical_column
        # Default dtype should be string
        if hasattr(cat_col_default.dtype, 'name'):
            assert cat_col_default.dtype.name == 'string'
        else:
            assert str(cat_col_default.dtype) == 'string'
        
        # Test error case: empty vocabulary list
        with pytest.raises(ValueError) as exc_info:
            sequence_categorical_column_with_vocabulary_list(
                key="test_feature",
                vocabulary_list=[]
            )
        
        # Check error message mentions vocabulary
        error_msg = str(exc_info.value).lower()
        assert 'vocabulary' in error_msg or 'empty' in error_msg
        
        # Test error case: duplicate values in vocabulary list
        with pytest.raises(ValueError) as exc_info:
            sequence_categorical_column_with_vocabulary_list(
                key="test_feature",
                vocabulary_list=["cat", "dog", "cat"]  # Duplicate "cat"
            )
        
        # Check error message mentions duplicate
        error_msg = str(exc_info.value).lower()
        assert 'duplicate' in error_msg
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:FOOTER START ====
# Additional helper functions and cleanup
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====