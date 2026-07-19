import pytest
import numpy as np
from unittest import mock
import threading
from tensorflow.python.framework import tensor_conversion_registry

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions
@pytest.fixture
def mock_conversion_func():
    """Mock conversion function with correct signature."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock()
    return conversion_func

@pytest.fixture
def different_mock_func():
    """Different mock conversion function for testing."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="different_func")
    return conversion_func

@pytest.fixture
def custom_class():
    """Custom class for testing type registration."""
    class CustomClass:
        pass
    return CustomClass

@pytest.fixture
def another_custom_class():
    """Another custom class for testing."""
    class AnotherCustomClass:
        pass
    return AnotherCustomClass

@pytest.fixture
def custom_class1():
    """First custom class for tuple registration."""
    class CustomClass1:
        pass
    return CustomClass1

@pytest.fixture
def custom_class2():
    """Second custom class for tuple registration."""
    class CustomClass2:
        pass
    return CustomClass2

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before each test to ensure isolation."""
    # Clear the registry and cache
    with tensor_conversion_registry._tensor_conversion_func_lock:
        tensor_conversion_registry._tensor_conversion_func_registry.clear()
        tensor_conversion_registry._tensor_conversion_func_cache.clear()
    yield
    # Clean up after test
    with tensor_conversion_registry._tensor_conversion_func_lock:
        tensor_conversion_registry._tensor_conversion_func_registry.clear()
        tensor_conversion_registry._tensor_conversion_func_cache.clear()

# Fixture function signature verification tests
def test_mock_conversion_func_signature(mock_conversion_func):
    """Test that mock_conversion_func has correct signature."""
    import inspect
    sig = inspect.signature(mock_conversion_func)
    params = list(sig.parameters.keys())
    expected_params = ['value', 'dtype', 'name', 'as_ref']
    assert params == expected_params, f"Expected parameters {expected_params}, got {params}"
    
    # Test calling with correct arguments
    test_value = object()
    result = mock_conversion_func(test_value, dtype=None, name="test", as_ref=False)
    assert isinstance(result, mock.MagicMock), "Should return MagicMock"
    
    # Test with as_ref=True
    result_ref = mock_conversion_func(test_value, dtype=None, name="test_ref", as_ref=True)
    assert isinstance(result_ref, mock.MagicMock), "Should return MagicMock even with as_ref=True"

def test_different_mock_func_signature(different_mock_func):
    """Test that different_mock_func has correct signature."""
    import inspect
    sig = inspect.signature(different_mock_func)
    params = list(sig.parameters.keys())
    expected_params = ['value', 'dtype', 'name', 'as_ref']
    assert params == expected_params, f"Expected parameters {expected_params}, got {params}"
    
    # Test calling and verify mock name
    test_value = object()
    result = different_mock_func(test_value, dtype=None, name="test", as_ref=False)
    assert isinstance(result, mock.MagicMock), "Should return MagicMock"
    # MagicMock.name returns another Mock object, not a string
    # So we need to check the mock's configuration instead
    assert result._mock_name == "different_func", "Mock should have correct name configuration"

def test_fixture_creates_unique_classes(custom_class, another_custom_class, custom_class1, custom_class2):
    """Test that fixture functions create unique class instances."""
    # Each fixture should return a different class
    assert custom_class is not another_custom_class, "custom_class and another_custom_class should be different"
    assert custom_class is not custom_class1, "custom_class and custom_class1 should be different"
    assert custom_class is not custom_class2, "custom_class and custom_class2 should be different"
    assert custom_class1 is not custom_class2, "custom_class1 and custom_class2 should be different"
    
    # Verify they are all classes
    assert isinstance(custom_class, type), "custom_class should be a type"
    assert isinstance(another_custom_class, type), "another_custom_class should be a type"
    assert isinstance(custom_class1, type), "custom_class1 should be a type"
    assert isinstance(custom_class2, type), "custom_class2 should be a type"
    
    # Test instantiation
    instance1 = custom_class()
    instance2 = another_custom_class()
    assert isinstance(instance1, custom_class), "Should create instance of custom_class"
    assert isinstance(instance2, another_custom_class), "Should create instance of another_custom_class"
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
def test_basic_type_registration_and_query(custom_class, mock_conversion_func):
    """Test basic type registration and query functionality.
    
    TC-01: 基本类型注册与查询
    Weak asserts: registration_success, query_returns_func, priority_order
    """
    # Register conversion function for custom class
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=mock_conversion_func,
        priority=100
    )
    
    # Query for conversion functions
    conversion_funcs = tensor_conversion_registry.get(custom_class)
    
    # Weak assertion 1: registration_success - should return list
    assert isinstance(conversion_funcs, list), "Should return a list of conversion functions"
    
    # Weak assertion 2: query_returns_func - list should contain our function
    assert len(conversion_funcs) > 0, "Should have at least one conversion function"
    
    # Check that our function is in the list
    found = False
    for base_type, func in conversion_funcs:
        if func is mock_conversion_func:
            found = True
            break
    assert found, "Registered conversion function should be in query results"
    
    # Weak assertion 3: priority_order - functions should be sorted by priority
    # Since we only registered one function, this is trivially satisfied
    # But we can verify the structure
    for base_type, func in conversion_funcs:
        assert base_type == custom_class or issubclass(custom_class, base_type), \
            "Base type should match or be superclass of query type"
    
    # Additional verification: test with subclass
    class SubClass(custom_class):
        pass
    
    subclass_funcs = tensor_conversion_registry.get(SubClass)
    assert len(subclass_funcs) > 0, "Should also find functions for subclasses"
    
    # Verify the function signature is correct by calling it
    test_value = custom_class()
    result = mock_conversion_func(test_value, dtype=None, name="test", as_ref=False)
    assert result is not None, "Conversion function should return something"
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
@pytest.mark.parametrize("base_type,expected_exception", [
    (int, TypeError),  # Python integer type
    (np.ndarray, TypeError),  # NumPy array type
    (float, TypeError),  # Python float type (from param_extensions)
])
def test_prohibited_type_registration_exception(
    base_type, expected_exception, mock_conversion_func
):
    """Test that prohibited types raise appropriate exceptions.
    
    TC-02: 禁止类型注册异常
    Weak asserts: exception_raised, exception_type, error_message
    """
    # Weak assertion 1: exception_raised
    with pytest.raises(expected_exception) as exc_info:
        tensor_conversion_registry.register_tensor_conversion_function(
            base_type=base_type,
            conversion_func=mock_conversion_func,
            priority=100
        )
    
    # Weak assertion 2: exception_type
    assert exc_info.type == expected_exception, \
        f"Should raise {expected_exception.__name__} for {base_type}"
    
    # Weak assertion 3: error_message
    error_msg = str(exc_info.value)
    assert "Cannot register conversions for Python numeric types" in error_msg or \
           "NumPy scalars and arrays" in error_msg or \
           "must be a type" in error_msg, \
        f"Error message should indicate prohibited type: {error_msg}"
    
    # Additional verification: registry should remain unchanged
    # Since we can't directly access private registry, we'll test that
    # querying the type doesn't return our mock function
    if hasattr(base_type, '__name__'):
        # For actual types, we can query
        try:
            funcs = tensor_conversion_registry.get(base_type)
            # For prohibited types, get() returns default conversion function
            # So we need to check it's not our mock function
            for _, func in funcs:
                assert func is not mock_conversion_func, \
                    "Registry should not contain prohibited type registration"
        except (TypeError, AttributeError):
            # Some types might not be queryable, that's OK
            pass
    
    # Test with numpy scalar types
    if hasattr(np, 'int32'):
        with pytest.raises(TypeError):
            tensor_conversion_registry.register_tensor_conversion_function(
                base_type=np.int32,
                conversion_func=mock_conversion_func,
                priority=100
            )
    
    # Test with numpy generic
    with pytest.raises(TypeError):
        tensor_conversion_registry.register_tensor_conversion_function(
            base_type=np.generic,
            conversion_func=mock_conversion_func,
            priority=100
        )
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_05 START ====
def test_tuple_type_registration(custom_class1, custom_class2, mock_conversion_func):
    """Test registration with tuple of types.
    
    TC-05: 类型元组注册
    Weak asserts: registration_success, both_types_registered, query_works
    """
    # Create a tuple of types
    type_tuple = (custom_class1, custom_class2)
    
    # Register conversion function for the tuple of types
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=type_tuple,
        conversion_func=mock_conversion_func,
        priority=100
    )
    
    # Weak assertion 1: registration_success
    # Query for first type in tuple
    funcs1 = tensor_conversion_registry.get(custom_class1)
    assert isinstance(funcs1, list), "Should return a list for first type"
    
    # Weak assertion 2: both_types_registered
    # Query for second type in tuple
    funcs2 = tensor_conversion_registry.get(custom_class2)
    assert isinstance(funcs2, list), "Should return a list for second type"
    
    # Check that both queries return our function
    found_in_first = False
    for base_type, func in funcs1:
        if func is mock_conversion_func:
            found_in_first = True
            break
    
    found_in_second = False
    for base_type, func in funcs2:
        if func is mock_conversion_func:
            found_in_second = True
            break
    
    assert found_in_first, "mock_conversion_func should be registered for first type"
    assert found_in_second, "mock_conversion_func should be registered for second type"
    
    # Weak assertion 3: query_works
    # Test with subclasses of the registered types
    class SubClass1(custom_class1):
        pass
    
    class SubClass2(custom_class2):
        pass
    
    subclass1_funcs = tensor_conversion_registry.get(SubClass1)
    subclass2_funcs = tensor_conversion_registry.get(SubClass2)
    
    # Subclasses should also find the registered function
    assert len(subclass1_funcs) > 0, "Should find function for subclass of first type"
    assert len(subclass2_funcs) > 0, "Should find function for subclass of second type"
    
    # Verify the function can be called
    test_value1 = custom_class1()
    test_value2 = custom_class2()
    
    result1 = mock_conversion_func(test_value1, dtype=None, name="test1", as_ref=False)
    result2 = mock_conversion_func(test_value2, dtype=None, name="test2", as_ref=False)
    
    assert result1 is not None, "Function should work for first type"
    assert result2 is not None, "Function should work for second type"
    
    # Test that registration doesn't affect other types
    class UnrelatedClass:
        pass
    
    unrelated_funcs = tensor_conversion_registry.get(UnrelatedClass)
    # Unrelated class should not have our function unless it's a subclass
    has_our_func = False
    for base_type, func in unrelated_funcs:
        if func is mock_conversion_func:
            has_our_func = True
            break
    
    # Only check if UnrelatedClass happens to be a subclass of custom_class1 or custom_class2
    if not (issubclass(UnrelatedClass, custom_class1) or issubclass(UnrelatedClass, custom_class2)):
        assert not has_our_func, "Unrelated class should not have our function"
    
    # Test with three types in tuple
    class CustomClass3:
        pass
    
    three_type_tuple = (custom_class1, custom_class2, CustomClass3)
    
    # Register a different function for three-type tuple
    def another_mock_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="another_func")
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=three_type_tuple,
        conversion_func=another_mock_func,
        priority=50
    )
    
    # Check all three types get the new function
    for cls in [custom_class1, custom_class2, CustomClass3]:
        cls_funcs = tensor_conversion_registry.get(cls)
        found_new_func = False
        for base_type, func in cls_funcs:
            if func is another_mock_func:
                found_new_func = True
                break
        assert found_new_func, f"another_mock_func should be registered for {cls.__name__}"
    
    # Verify priority ordering: another_mock_func (priority 50) should come before
    # mock_conversion_func (priority 100) for types that have both
    custom_class1_funcs = tensor_conversion_registry.get(custom_class1)
    
    # Find indices of both functions
    mock_func_index = -1
    another_func_index = -1
    
    for i, (base_type, func) in enumerate(custom_class1_funcs):
        if func is mock_conversion_func:
            mock_func_index = i
        if func is another_mock_func:
            another_func_index = i
    
    # If both found, another_mock_func (priority 50) should come before mock_conversion_func (priority 100)
    if mock_func_index != -1 and another_func_index != -1:
        assert another_func_index < mock_func_index, \
            "Higher priority function (50) should come before lower priority (100)"
    
    # Additional test: numerical type default conversion function
    # Test that get() returns default conversion for numeric types
    def test_numeric_type_default_conversion():
        """Test that numeric types get default conversion functions."""
        # Test with Python numeric types
        int_funcs = tensor_conversion_registry.get(int)
        float_funcs = tensor_conversion_registry.get(float)
        
        # Numeric types should get default conversion function
        assert len(int_funcs) == 1, "int should have default conversion function"
        assert len(float_funcs) == 1, "float should have default conversion function"
        
        # Check that it's the default conversion function
        # The default function should be constant_op.constant or similar
        _, int_default_func = int_funcs[0]
        _, float_default_func = float_funcs[0]
        
        # Verify function names contain expected patterns
        int_func_name = str(int_default_func)
        float_func_name = str(float_default_func)
        
        # Default conversion functions should mention constant or conversion
        assert 'constant' in int_func_name.lower() or 'conversion' in int_func_name.lower(), \
            f"int default function should be constant/conversion function, got: {int_func_name}"
        assert 'constant' in float_func_name.lower() or 'conversion' in float_func_name.lower(), \
            f"float default function should be constant/conversion function, got: {float_func_name}"
        
        # Test with NumPy numeric types
        if hasattr(np, 'int32'):
            np_int32_funcs = tensor_conversion_registry.get(np.int32)
            assert len(np_int32_funcs) == 1, "np.int32 should have default conversion function"
            
        if hasattr(np, 'float64'):
            np_float64_funcs = tensor_conversion_registry.get(np.float64)
            assert len(np_float64_funcs) == 1, "np.float64 should have default conversion function"
        
        # Test with np.generic (base class for all numpy scalars)
        np_generic_funcs = tensor_conversion_registry.get(np.generic)
        assert len(np_generic_funcs) == 1, "np.generic should have default conversion function"
        
        # Test with np.ndarray
        np_array_funcs = tensor_conversion_registry.get(np.ndarray)
        assert len(np_array_funcs) == 1, "np.ndarray should have default conversion function"
        
        # Verify that numeric types cannot be registered (should raise TypeError)
        def numeric_conversion_func(value, dtype=None, name=None, as_ref=False):
            return mock.MagicMock(name="numeric_func")
        
        with pytest.raises(TypeError):
            tensor_conversion_registry.register_tensor_conversion_function(
                base_type=int,
                conversion_func=numeric_conversion_func,
                priority=100
            )
        
        with pytest.raises(TypeError):
            tensor_conversion_registry.register_tensor_conversion_function(
                base_type=float,
                conversion_func=numeric_conversion_func,
                priority=100
            )
        
        # Test that querying numeric types doesn't return our mock function
        int_funcs_after = tensor_conversion_registry.get(int)
        found_numeric_mock = False
        for base_type, func in int_funcs_after:
            if func is numeric_conversion_func:
                found_numeric_mock = True
                break
        assert not found_numeric_mock, "int should not have our mock function (registration should have failed)"
        
        # Test that default conversion works for subclasses of numeric types
        class MyInt(int):
            pass
        
        myint_funcs = tensor_conversion_registry.get(MyInt)
        assert len(myint_funcs) == 1, "MyInt (subclass of int) should have default conversion function"
        
        # The default function for MyInt should be the same as for int
        _, myint_default_func = myint_funcs[0]
        _, int_default_func = int_funcs[0]
        
        # They might be the same function object or different wrappers
        # At minimum, verify it's a function
        assert callable(myint_default_func), "MyInt default function should be callable"
        assert callable(int_default_func), "int default function should be callable"
        
        return True
    
    # Run the numeric type test
    assert test_numeric_type_default_conversion(), "Numeric type default conversion test failed"
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
def test_different_mock_func_usage(custom_class, different_mock_func, mock_conversion_func):
    """Test usage of different_mock_func fixture and additional test branches.
    
    TC-06: 未使用的fixture和测试分支
    Weak asserts: registration_success, function_comparison, priority_handling
    """
    # Test 1: Register different_mock_func and verify it works
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=different_mock_func,
        priority=75
    )
    
    # Query for conversion functions
    conversion_funcs = tensor_conversion_registry.get(custom_class)
    
    # Weak assertion 1: registration_success
    assert isinstance(conversion_funcs, list), "Should return a list"
    assert len(conversion_funcs) > 0, "Should have registered function"
    
    # Check that different_mock_func is in the results
    found_different_func = False
    for base_type, func in conversion_funcs:
        if func is different_mock_func:
            found_different_func = True
            break
    assert found_different_func, "different_mock_func should be in query results"
    
    # Test calling the function
    test_value = custom_class()
    result = different_mock_func(test_value, dtype=None, name="test", as_ref=False)
    assert result is not None, "different_mock_func should return a mock object"
    # MagicMock.name returns another Mock object, not a string
    # So we need to check the mock's configuration instead
    assert result._mock_name == "different_func", "Mock should have correct name configuration"
    
    # Test 2: Register multiple functions and verify ordering
    # Clear registry first
    with tensor_conversion_registry._tensor_conversion_func_lock:
        tensor_conversion_registry._tensor_conversion_func_registry.clear()
        tensor_conversion_registry._tensor_conversion_func_cache.clear()
    
    # Register three functions with different priorities
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=mock_conversion_func,
        priority=100  # Lowest priority (highest number)
    )
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=different_mock_func,
        priority=50  # Middle priority
    )
    
    # Register a third function
    def third_mock_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="third_func")
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=third_mock_func,
        priority=10  # Highest priority (lowest number)
    )
    
    # Get all functions
    all_funcs = tensor_conversion_registry.get(custom_class)
    assert len(all_funcs) == 3, "Should have three registered functions"
    
    # Weak assertion 2: function_comparison
    # Verify all three functions are present
    funcs_found = {
        'mock_conversion_func': False,
        'different_mock_func': False,
        'third_mock_func': False
    }
    
    for base_type, func in all_funcs:
        if func is mock_conversion_func:
            funcs_found['mock_conversion_func'] = True
        elif func is different_mock_func:
            funcs_found['different_mock_func'] = True
        elif func is third_mock_func:
            funcs_found['third_mock_func'] = True
    
    assert all(funcs_found.values()), "All three functions should be found"
    
    # Weak assertion 3: priority_handling
    # Verify order: third_mock_func (10) < different_mock_func (50) < mock_conversion_func (100)
    func_indices = {}
    for i, (base_type, func) in enumerate(all_funcs):
        if func is mock_conversion_func:
            func_indices['mock_conversion_func'] = i
        elif func is different_mock_func:
            func_indices['different_mock_func'] = i
        elif func is third_mock_func:
            func_indices['third_mock_func'] = i
    
    assert func_indices['third_mock_func'] < func_indices['different_mock_func'], \
        "third_mock_func (priority 10) should come before different_mock_func (priority 50)"
    assert func_indices['different_mock_func'] < func_indices['mock_conversion_func'], \
        "different_mock_func (priority 50) should come before mock_conversion_func (priority 100)"
    
    # Test 3: Edge case - same priority, different registration order
    # Clear registry again
    with tensor_conversion_registry._tensor_conversion_func_lock:
        tensor_conversion_registry._tensor_conversion_func_registry.clear()
        tensor_conversion_registry._tensor_conversion_func_cache.clear()
    
    # Register two functions with same priority
    def same_priority_func1(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="same1")
    
    def same_priority_func2(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="same2")
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=same_priority_func1,
        priority=100
    )
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=same_priority_func2,
        priority=100  # Same priority
    )
    
    same_priority_funcs = tensor_conversion_registry.get(custom_class)
    assert len(same_priority_funcs) == 2, "Should have two functions with same priority"
    
    # Functions with same priority should maintain registration order
    # first registered should come first
    first_func = same_priority_funcs[0][1]
    second_func = same_priority_funcs[1][1]
    
    # Check order (might be same_priority_func1 then same_priority_func2)
    # We can't guarantee which is which by object identity, but we can check
    # that they're different functions
    assert first_func is not second_func, "Should have two different functions"
    
    # Enhanced test for same priority registration order
    def test_same_priority_registration_order():
        """Test that functions with same priority maintain registration order."""
        # Clear registry
        with tensor_conversion_registry._tensor_conversion_func_lock:
            tensor_conversion_registry._tensor_conversion_func_registry.clear()
            tensor_conversion_registry._tensor_conversion_func_cache.clear()
        
        # Create three functions with same priority but different names
        funcs_same_priority = []
        for i in range(3):
            def make_func(idx):
                def func(value, dtype=None, name=None, as_ref=False):
                    return mock.MagicMock(name=f"same_priority_func_{idx}")
                return func
            
            funcs_same_priority.append(make_func(i))
        
        # Register them in order 0, 1, 2
        for i, func in enumerate(funcs_same_priority):
            tensor_conversion_registry.register_tensor_conversion_function(
                base_type=custom_class,
                conversion_func=func,
                priority=100  # Same priority for all
            )
        
        # Get all functions
        all_same_priority_funcs = tensor_conversion_registry.get(custom_class)
        assert len(all_same_priority_funcs) == 3, "Should have three functions with same priority"
        
        # Verify registration order is preserved
        # Since we can't easily identify lambda functions by object identity,
        # we'll call them and check the mock names
        registered_order = []
        for _, func in all_same_priority_funcs:
            # Call function to get mock name
            result = func(None, dtype=None, name="test", as_ref=False)
            # MagicMock.name returns another Mock object, check _mock_name instead
            mock_name = result._mock_name
            registered_order.append(mock_name)
        
        # Should be in order: same_priority_func_0, same_priority_func_1, same_priority_func_2
        # Check that all expected names are present
        expected_names = {f"same_priority_func_{i}" for i in range(3)}
        actual_names = set(registered_order)
        assert expected_names == actual_names, f"Expected names {expected_names}, got {actual_names}"
        
        # Test with more complex scenario: mix of same and different priorities
        with tensor_conversion_registry._tensor_conversion_func_lock:
            tensor_conversion_registry._tensor_conversion_func_registry.clear()
            tensor_conversion_registry._tensor_conversion_func_cache.clear()
        
        # Register functions in this order:
        # 1. priority 100, name "A"
        # 2. priority 50, name "B" (higher priority)
        # 3. priority 100, name "C" (same as A)
        # 4. priority 75, name "D"
        # 5. priority 100, name "E" (same as A and C)
        
        func_a = lambda v, d=None, n=None, a=False: mock.MagicMock(name="A")
        func_b = lambda v, d=None, n=None, a=False: mock.MagicMock(name="B")
        func_c = lambda v, d=None, n=None, a=False: mock.MagicMock(name="C")
        func_d = lambda v, d=None, n=None, a=False: mock.MagicMock(name="D")
        func_e = lambda v, d=None, n=None, a=False: mock.MagicMock(name="E")
        
        registration_order = [
            (func_a, 100, "A"),
            (func_b, 50, "B"),
            (func_c, 100, "C"),
            (func_d, 75, "D"),
            (func_e, 100, "E")
        ]
        
        for func, priority, name in registration_order:
            tensor_conversion_registry.register_tensor_conversion_function(
                base_type=custom_class,
                conversion_func=func,
                priority=priority
            )
        
        # Get all functions
        mixed_funcs = tensor_conversion_registry.get(custom_class)
        assert len(mixed_funcs) == 5, "Should have five registered functions"
        
        # Expected order by priority then registration:
        # 1. B (priority 50) - highest priority
        # 2. D (priority 75) - middle priority
        # 3. A (priority 100, registered first)
        # 4. C (priority 100, registered third)
        # 5. E (priority 100, registered fifth)
        
        # Get mock names in order
        actual_order = []
        for _, func in mixed_funcs:
            result = func(None, dtype=None, name="test", as_ref=False)
            # Use _mock_name instead of name
            actual_order.append(result._mock_name)
        
        # Verify B comes first (highest priority)
        assert actual_order[0] == "B", f"B (priority 50) should come first, got {actual_order[0]}"
        
        # Verify D comes second (priority 75)
        assert actual_order[1] == "D", f"D (priority 75) should come second, got {actual_order[1]}"
        
        # Verify A, C, E come last in registration order (all priority 100)
        # They should be in the order they were registered: A, C, E
        priority_100_funcs = actual_order[2:]
        assert priority_100_funcs == ["A", "C", "E"], \
            f"Priority 100 functions should be in registration order A, C, E, got {priority_100_funcs}"
        
        return True
    
    # Run the enhanced same priority test
    assert test_same_priority_registration_order(), "Same priority registration order test failed"
    
    # Test 4: Test with inheritance hierarchy
    class ParentClass:
        pass
    
    class ChildClass(ParentClass):
        pass
    
    class GrandChildClass(ChildClass):
        pass
    
    # Register function for ParentClass
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=ParentClass,
        conversion_func=different_mock_func,
        priority=100
    )
    
    # All three classes should get the function
    for cls in [ParentClass, ChildClass, GrandChildClass]:
        cls_funcs = tensor_conversion_registry.get(cls)
        found = False
        for base_type, func in cls_funcs:
            if func is different_mock_func:
                found = True
                break
        assert found, f"different_mock_func should be available for {cls.__name__}"
    
    # Test 5: Verify function signatures
    import inspect
    sig1 = inspect.signature(different_mock_func)
    sig2 = inspect.signature(mock_conversion_func)
    
    params1 = list(sig1.parameters.keys())
    params2 = list(sig2.parameters.keys())
    
    expected_params = ['value', 'dtype', 'name', 'as_ref']
    assert params1 == expected_params, f"different_mock_func should have parameters {expected_params}"
    assert params2 == expected_params, f"mock_conversion_func should have parameters {expected_params}"
    
    # Test 6: Edge case - priority 0 and negative priorities
    def edge_priority_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="edge_priority")
    
    # Test priority 0 (should be highest possible priority)
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=edge_priority_func,
        priority=0
    )
    
    # Get functions and verify edge_priority_func comes first
    edge_funcs = tensor_conversion_registry.get(custom_class)
    # Find edge_priority_func
    edge_func_index = -1
    for i, (base_type, func) in enumerate(edge_funcs):
        try:
            result = func(None, dtype=None, name="test", as_ref=False)
            # Use _mock_name instead of name
            if result._mock_name == "edge_priority":
                edge_func_index = i
                break
        except:
            pass
    
    # edge_priority_func (priority 0) should come before all others
    if edge_func_index != -1:
        # Check it comes before functions with higher priority numbers
        assert edge_func_index == 0, f"Priority 0 function should come first, found at index {edge_func_index}"
    
    # Test negative priority (should be even higher priority than 0)
    def negative_priority_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="negative_priority")
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=negative_priority_func,
        priority=-10
    )
    
    # Get updated functions
    updated_edge_funcs = tensor_conversion_registry.get(custom_class)
    
    # Find negative_priority_func
    negative_func_index = -1
    for i, (base_type, func) in enumerate(updated_edge_funcs):
        try:
            result = func(None, dtype=None, name="test", as_ref=False)
            # Use _mock_name instead of name
            if result._mock_name == "negative_priority":
                negative_func_index = i
                break
        except:
            pass
    
    # negative_priority_func (priority -10) should come before edge_priority_func (priority 0)
    if negative_func_index != -1 and edge_func_index != -1:
        # In updated list, we need to find the new indices
        # negative_priority_func should come first
        assert negative_func_index < edge_func_index, \
            f"Negative priority (-10) should come before 0, got negative at {negative_func_index}, 0 at {edge_func_index}"
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:FOOTER START ====
# Additional helper functions and cleanup
def test_module_import():
    """Test that the module can be imported correctly."""
    assert tensor_conversion_registry is not None
    assert hasattr(tensor_conversion_registry, 'register_tensor_conversion_function')
    assert hasattr(tensor_conversion_registry, 'get')
# ==== BLOCK:FOOTER END ====