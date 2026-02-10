import pytest
import tensorflow as tf
from tensorflow.python.util import nest


# ==== BLOCK:HEADER START ====
"""
Test cases for tensorflow.python.util.nest module.

This module tests the nested structure manipulation functions including:
- flatten: flatten nested structures into a flat list
- map_structure: apply a function to each element while preserving structure
- assert_same_structure: validate that two structures have the same shape
- pack_sequence_as: rebuild a structure from a flat sequence
- is_nested: check if an object is a nested structure

All tests use weak assertions in the first round (epoch=1).
"""
# ==== BLOCK:HEADER END ====


# ==== BLOCK:CASE_01 START ====
class TestFlattenBasic:
    """Test cases for tf.nest.flatten function with basic nested structures."""
    
    @pytest.mark.parametrize(
        "structure, expected, description",
        [
            ([[1, 2], [3, 4]], [1, 2, 3, 4], "列表嵌套列表"),
            ({"a": 1, "b": 2}, [1, 2], "简单字典"),
            (([1, 2], {"x": 3}), [1, 2, 3], "元组和字典混合"),
            ([], [], "空列表"),
            ({}, [], "空字典"),
        ]
    )
    def test_flatten_basic_structures(self, structure, expected, description):
        """Test flatten with various basic nested structures."""
        # Act
        result = nest.flatten(structure)
        
        # Assert - weak assertions
        # 1. Check length matches expected
        assert len(result) == len(expected), (
            f"Length mismatch for {description}: "
            f"expected {len(expected)}, got {len(result)}"
        )
        
        # 2. Check element order (weak check - just compare elements)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res == exp, (
                f"Element mismatch at position {i} for {description}: "
                f"expected {exp}, got {res}"
            )
        
        # 3. Basic property check: result should be a list
        assert isinstance(result, list), (
            f"Result should be a list for {description}, got {type(result)}"
        )
        
        # Strong assertions (final round)
        # 1. Exact match: result should exactly equal expected
        assert result == expected, (
            f"Exact match failed for {description}: "
            f"expected {expected}, got {result}"
        )
        
        # 2. Type preservation: check specific element types if applicable
        if result and expected:
            for i, (res, exp) in enumerate(zip(result, expected)):
                # For numeric types, check type consistency
                if isinstance(exp, (int, float)):
                    assert isinstance(res, type(exp)), (
                        f"Type mismatch at position {i} for {description}: "
                        f"expected type {type(exp)}, got {type(res)}"
                    )
        
        # 3. Edge cases: additional checks for boundary conditions
        if not result and not expected:  # Both empty
            # Verify empty list properties
            assert result == [], f"Empty result should be [] for {description}"
            assert len(result) == 0, f"Empty result length should be 0 for {description}"
        
        # Additional strong check: verify flatten is idempotent for flat structures
        if not nest.is_nested(structure):
            # For non-nested structures, flatten should return a single-element list
            flat_result = nest.flatten(structure)
            assert len(flat_result) == 1, (
                f"Flatten of non-nested structure should have length 1 for {description}"
            )
            assert flat_result[0] == structure, (
                f"Flatten of non-nested structure should preserve value for {description}"
            )
# ==== BLOCK:CASE_01 END ====


# ==== BLOCK:CASE_02 START ====
class TestMapStructure:
    """Test cases for tf.nest.map_structure function."""
    
    @pytest.mark.parametrize(
        "func, structure, expected, description",
        [
            (lambda x: x * 2, {"a": 1, "b": 2}, {"a": 2, "b": 4}, "字典乘以2"),
            (lambda x: x + 1, [1, [2, 3]], [2, [3, 4]], "嵌套列表加1"),
            (str, (1, 2, 3), ("1", "2", "3"), "元组元素转字符串"),
            (lambda x: None, [1, 2, 3], [None, None, None], "函数返回None"),
        ]
    )
    def test_map_structure_basic(self, func, structure, expected, description):
        """Test map_structure with various functions and structures."""
        # Act
        result = nest.map_structure(func, structure)
        
        # Assert - weak assertions
        # 1. Check structure is preserved (same type)
        assert type(result) == type(expected), (
            f"Structure type mismatch for {description}: "
            f"expected {type(expected)}, got {type(result)}"
        )
        
        # 2. Check function was applied correctly (weak check)
        # For dicts, check keys and values
        if isinstance(expected, dict):
            assert set(result.keys()) == set(expected.keys()), (
                f"Dictionary keys mismatch for {description}"
            )
            for key in expected:
                assert result[key] == expected[key], (
                    f"Value mismatch for key '{key}' in {description}: "
                    f"expected {expected[key]}, got {result[key]}"
                )
        
        # 3. For lists/tuples, check length and elements
        elif isinstance(expected, (list, tuple)):
            assert len(result) == len(expected), (
                f"Length mismatch for {description}: "
                f"expected {len(expected)}, got {len(result)}"
            )
            for i, (res, exp) in enumerate(zip(result, expected)):
                assert res == exp, (
                    f"Element mismatch at position {i} for {description}: "
                    f"expected {exp}, got {res}"
                )
        
        # 4. Basic property: result should have same structure as input
        # We can check by flattening both and comparing lengths
        flat_result = nest.flatten(result)
        flat_expected = nest.flatten(expected)
        assert len(flat_result) == len(flat_expected), (
            f"Flattened length mismatch for {description}: "
            f"expected {len(flat_expected)}, got {len(flat_result)}"
        )
        
        # Strong assertions (final round)
        # 1. Exact structure match: deep equality check
        # Use nest.assert_same_structure for structural equality
        try:
            nest.assert_same_structure(result, expected, check_types=True)
        except ValueError as e:
            pytest.fail(
                f"Exact structure match failed for {description}: {e}"
            )
        
        # 2. Function composition: verify map_structure with identity function
        identity = lambda x: x
        identity_result = nest.map_structure(identity, structure)
        
        # Result should be structurally identical to input
        try:
            nest.assert_same_structure(identity_result, structure, check_types=True)
        except ValueError as e:
            pytest.fail(
                f"Identity function composition failed for {description}: {e}"
            )
        
        # 3. Error handling: test with invalid inputs
        # Test that map_structure raises appropriate errors for mismatched structures
        if isinstance(structure, dict) and structure:
            # Create a function that might raise an error
            def raising_func(x):
                if x == 0:
                    raise ValueError("Zero not allowed")
                return x * 2
            
            # Test with a structure that won't trigger the error
            safe_structure = {k: v + 1 for k, v in structure.items()}  # Ensure no zeros
            safe_result = nest.map_structure(raising_func, safe_structure)
            
            # Verify the function was applied correctly
            for key in safe_structure:
                expected_val = raising_func(safe_structure[key])
                assert safe_result[key] == expected_val, (
                    f"Error handling test failed for key '{key}' in {description}: "
                    f"expected {expected_val}, got {safe_result[key]}"
                )
        
        # Additional strong check: verify structure preservation with nested flatten
        original_flat = nest.flatten(structure)
        result_flat = nest.flatten(result)
        
        # The number of elements should be preserved
        assert len(original_flat) == len(result_flat), (
            f"Element count not preserved for {description}: "
            f"original had {len(original_flat)}, result has {len(result_flat)}"
        )
        
        # Verify function was applied to each element
        for i, (orig, res) in enumerate(zip(original_flat, result_flat)):
            expected_val = func(orig)
            assert res == expected_val, (
                f"Function application mismatch at flattened position {i} for {description}: "
                f"expected {expected_val} (func({orig})), got {res}"
            )
# ==== BLOCK:CASE_02 END ====


# ==== BLOCK:CASE_03 START ====
class TestAssertSameStructure:
    """Test cases for tf.nest.assert_same_structure function."""
    
    @pytest.mark.parametrize(
        "struct1, struct2, should_pass, check_types, description",
        [
            ([1, 2, 3], ["a", "b", "c"], True, False, "相同结构不同类型"),
            ({"x": 1, "y": 2}, {"x": 3, "y": 4}, True, True, "相同字典结构"),
            ([1, [2, 3]], [4, [5, 6]], True, True, "嵌套列表结构"),
            ([1, 2], [1, 2, 3], False, True, "不同长度结构"),
        ]
    )
    def test_assert_same_structure(self, struct1, struct2, should_pass, check_types, description):
        """Test assert_same_structure with various scenarios."""
        
        if should_pass:
            # Should not raise exception
            try:
                nest.assert_same_structure(struct1, struct2, check_types=check_types)
                # If we get here, assertion passed as expected
                assert True, f"assert_same_structure should pass for {description}"
            except ValueError as e:
                pytest.fail(f"assert_same_structure raised ValueError for {description}: {e}")
        else:
            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                nest.assert_same_structure(struct1, struct2, check_types=check_types)
            
            # Weak assertion: just check that an exception was raised
            assert exc_info.type == ValueError, (
                f"Expected ValueError for {description}, got {exc_info.type}"
            )
            
            # Basic validation: exception message should not be empty
            assert str(exc_info.value), (
                f"Exception message should not be empty for {description}"
            )
        
        # Strong assertions (final round)
        # 1. Exception type: verify exact exception type for failure cases
        if not should_pass:
            # Re-run to capture exception details
            with pytest.raises(ValueError) as exc_info:
                nest.assert_same_structure(struct1, struct2, check_types=check_types)
            
            # Verify it's specifically a ValueError (not TypeError or other)
            assert exc_info.type == ValueError, (
                f"Expected exact exception type ValueError for {description}, "
                f"got {exc_info.type}"
            )
            
            # 2. Error message: check error message contains useful information
            error_msg = str(exc_info.value).lower()
            
            # Check for relevant keywords in error message
            relevant_terms = ["structure", "length", "different", "mismatch"]
            has_relevant_term = any(term in error_msg for term in relevant_terms)
            assert has_relevant_term, (
                f"Error message should contain relevant structural information for {description}. "
                f"Message: {error_msg}"
            )
            
            # 3. Type checking: verify check_types parameter works correctly
            # For failure cases, we should still test check_types parameter
            # by verifying that the same error occurs regardless of check_types value
            with pytest.raises(ValueError) as exc_info_false:
                nest.assert_same_structure(struct1, struct2, check_types=False)
            
            with pytest.raises(ValueError) as exc_info_true:
                nest.assert_same_structure(struct1, struct2, check_types=True)
            
            # Both should raise ValueError for structural mismatch
            assert exc_info_false.type == ValueError
            assert exc_info_true.type == ValueError
            
            # Additional test: verify symmetry
            # assert_same_structure should be symmetric - both directions should fail
            with pytest.raises(ValueError) as exc_info_reverse:
                nest.assert_same_structure(struct2, struct1, check_types=check_types)
            
            assert exc_info_reverse.type == ValueError, (
                f"assert_same_structure should also fail in reverse direction for {description}"
            )
        
        # Additional strong checks for passing cases
        if should_pass:
            # Verify that flatten produces same number of elements
            flat1 = nest.flatten(struct1)
            flat2 = nest.flatten(struct2)
            
            assert len(flat1) == len(flat2), (
                f"Flattened structures should have same length for {description}: "
                f"struct1 has {len(flat1)} elements, struct2 has {len(flat2)} elements"
            )
            
            # Verify structure depth is preserved
            def max_depth(obj):
                if not nest.is_nested(obj):
                    return 0
                if isinstance(obj, dict):
                    if not obj:
                        return 1
                    return 1 + max(max_depth(v) for v in obj.values())
                if isinstance(obj, (list, tuple)):
                    if not obj:
                        return 1
                    return 1 + max(max_depth(item) for item in obj)
                return 1
            
            depth1 = max_depth(struct1)
            depth2 = max_depth(struct2)
            
            assert depth1 == depth2, (
                f"Structure depth should match for {description}: "
                f"struct1 depth={depth1}, struct2 depth={depth2}"
            )
            
            # Verify symmetry for passing cases
            try:
                nest.assert_same_structure(struct2, struct1, check_types=check_types)
            except ValueError as e:
                pytest.fail(
                    f"assert_same_structure is not symmetric for {description}: {e}"
                )
# ==== BLOCK:CASE_03 END ====


# ==== BLOCK:CASE_04 START ====
class TestPackSequenceAs:
    """Test cases for tf.nest.pack_sequence_as function."""
    
    @pytest.mark.parametrize(
        "structure, flat_sequence, expected, description",
        [
            ([0, 0], [1, 2], [1, 2], "简单列表重建"),
            ({"a": 0, "b": 0}, [3, 4], {"a": 3, "b": 4}, "字典重建"),
            ([1, [2, 3]], [10, 20, 30], [10, [20, 30]], "嵌套结构重建"),
            ((0, 0), [1, 2], (1, 2), "元组重建"),
        ]
    )
    def test_pack_sequence_as_basic(self, structure, flat_sequence, expected, description):
        """Test pack_sequence_as with various structures and sequences."""
        # Act
        result = nest.pack_sequence_as(structure, flat_sequence)
        
        # Assert - weak assertions
        # 1. Check structure matches expected type
        assert type(result) == type(expected), (
            f"Structure type mismatch for {description}: "
            f"expected {type(expected)}, got {type(result)}"
        )
        
        # 2. Check element order (weak check - compare flattened results)
        flat_result = nest.flatten(result)
        flat_expected = nest.flatten(expected)
        
        assert len(flat_result) == len(flat_expected), (
            f"Flattened length mismatch for {description}: "
            f"expected {len(flat_expected)}, got {len(flat_result)}"
        )
        
        for i, (res, exp) in enumerate(zip(flat_result, flat_expected)):
            assert res == exp, (
                f"Element mismatch at flattened position {i} for {description}: "
                f"expected {exp}, got {res}"
            )
        
        # 3. Basic property: result should have same structure as template
        # We can check by comparing the structure of result with input structure
        # using assert_same_structure (but with check_types=False for weak assertion)
        try:
            nest.assert_same_structure(result, structure, check_types=False)
        except ValueError as e:
            pytest.fail(
                f"Result structure doesn't match template for {description}: {e}"
            )
# ==== BLOCK:CASE_04 END ====


# ==== BLOCK:CASE_05 START ====
class TestIsNested:
    """Test cases for tf.nest.is_nested function."""
    
    @pytest.mark.parametrize(
        "structure, expected, description",
        [
            ([1, 2, 3], True, "列表是嵌套结构"),
            ({"a": 1}, True, "字典是嵌套结构"),
            (42, False, "整数不是嵌套结构"),
            ("string", False, "字符串不是嵌套结构"),
        ]
    )
    def test_is_nested_basic(self, structure, expected, description):
        """Test is_nested with various basic types."""
        # Act
        result = nest.is_nested(structure)
        
        # Assert - weak assertions
        # 1. Check boolean result matches expected
        assert result == expected, (
            f"is_nested result mismatch for {description}: "
            f"expected {expected}, got {result}"
        )
        
        # 2. Check basic type consistency
        assert isinstance(result, bool), (
            f"Result should be boolean for {description}, got {type(result)}"
        )
        
        # 3. Additional weak checks based on expected value
        if expected:
            # If expected True, structure should be a recognized nested type
            # Weak check: just verify it's not a simple scalar
            assert not isinstance(structure, (int, float, str, bytes)), (
                f"Structure {description} should not be a simple scalar "
                f"if is_nested returns True"
            )
        else:
            # If expected False, structure should be an atom
            # Weak check: verify it's a simple type or other non-nested type
            pass  # No specific weak assertion for False case
# ==== BLOCK:CASE_05 END ====


# ==== BLOCK:FOOTER START ====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# ==== BLOCK:FOOTER END ====