import pytest
import numpy as np
from unittest import mock
import threading
from tensorflow.python.framework import tensor_conversion_registry

# ==== BLOCK:HEADER START ====
# Test fixtures and helper functions for G2 group
@pytest.fixture
def mock_conversion_func():
    """Mock conversion function with correct signature."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock()
    return conversion_func

@pytest.fixture
def func_low():
    """Mock conversion function with low priority."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="func_low")
    return conversion_func

@pytest.fixture
def func_high():
    """Mock conversion function with high priority."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="func_high")
    return conversion_func

@pytest.fixture
def func1():
    """Mock conversion function 1."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="func1")
    return conversion_func

@pytest.fixture
def func2():
    """Mock conversion function 2."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="func2")
    return conversion_func

@pytest.fixture
def func3():
    """Mock conversion function 3."""
    def conversion_func(value, dtype=None, name=None, as_ref=False):
        return mock.MagicMock(name="func3")
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
def unregistered_class():
    """Class that is not registered."""
    class UnregisteredClass:
        pass
    return UnregisteredClass

@pytest.fixture
def another_unregistered_class():
    """Another class that is not registered."""
    class AnotherUnregisteredClass:
        pass
    return AnotherUnregisteredClass

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

# Helper function to test mock_conversion_func usage
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
    
    # Test with dtype specified
    result_with_dtype = mock_conversion_func(test_value, dtype="float32", name="test_dtype", as_ref=False)
    assert isinstance(result_with_dtype, mock.MagicMock), "Should return MagicMock with dtype"

# Test other fixture signatures
def test_func_low_signature(func_low):
    """Test that func_low has correct signature."""
    import inspect
    sig = inspect.signature(func_low)
    params = list(sig.parameters.keys())
    expected_params = ['value', 'dtype', 'name', 'as_ref']
    assert params == expected_params, f"Expected parameters {expected_params}, got {params}"
    
    # Test calling
    test_value = object()
    result = func_low(test_value, dtype=None, name="test", as_ref=False)
    assert isinstance(result, mock.MagicMock), "Should return MagicMock"
    assert result._mock_name == "func_low", "Mock should have correct name"

def test_func_high_signature(func_high):
    """Test that func_high has correct signature."""
    import inspect
    sig = inspect.signature(func_high)
    params = list(sig.parameters.keys())
    expected_params = ['value', 'dtype', 'name', 'as_ref']
    assert params == expected_params, f"Expected parameters {expected_params}, got {params}"
    
    # Test calling
    test_value = object()
    result = func_high(test_value, dtype=None, name="test", as_ref=False)
    assert isinstance(result, mock.MagicMock), "Should return MagicMock"
    assert result._mock_name == "func_high", "Mock should have correct name"
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_03 START ====
def test_query_unregistered_type(unregistered_class, another_unregistered_class, mock_conversion_func):
    """Test querying for unregistered types.
    
    TC-03: 查询未注册类型
    Weak asserts: returns_list, list_length, contains_default
    """
    # Test with first unregistered class
    # Weak assertion 1: returns_list
    conversion_funcs = tensor_conversion_registry.get(unregistered_class)
    assert isinstance(conversion_funcs, list), "Should return a list"
    
    # Weak assertion 2: list_length
    # For unregistered non-numeric types, should return empty list
    # For numeric types, get() returns default conversion function
    if issubclass(unregistered_class, (int, float, np.generic, np.ndarray)):
        # Numeric types get default conversion
        assert len(conversion_funcs) == 1, "Numeric types should get default conversion"
        # Weak assertion 3: contains_default
        _, func = conversion_funcs[0]
        # Check it's the default conversion function
        assert func.__name__ == '_default_conversion_function' or \
               'constant_op.constant' in str(func), \
            "Should return default conversion function for numeric types"
    else:
        # Non-numeric unregistered types return empty list
        assert len(conversion_funcs) == 0, "Unregistered non-numeric types should return empty list"
    
    # Test with another unregistered class (from param_extensions)
    conversion_funcs2 = tensor_conversion_registry.get(another_unregistered_class)
    assert isinstance(conversion_funcs2, list), "Should return a list for second class"
    
    # Verify consistency: querying same type twice should give same result
    conversion_funcs3 = tensor_conversion_registry.get(unregistered_class)
    assert conversion_funcs3 == conversion_funcs, "Multiple queries should return same result"
    
    # Test with a class that inherits from unregistered class
    class SubClass(unregistered_class):
        pass
    
    subclass_funcs = tensor_conversion_registry.get(SubClass)
    assert isinstance(subclass_funcs, list), "Should return list for subclass"
    
    # Test that querying doesn't modify registry
    # We can't directly check private registry, but we can verify
    # that subsequent queries still return same result
    conversion_funcs4 = tensor_conversion_registry.get(unregistered_class)
    assert conversion_funcs4 == conversion_funcs, "Registry should not be modified by query"
    
    # Test with built-in non-numeric type (e.g., str)
    str_funcs = tensor_conversion_registry.get(str)
    assert isinstance(str_funcs, list), "Should return list for str type"
    # str is not numeric, so should return empty list if not registered
    if not issubclass(str, (int, float, np.generic, np.ndarray)):
        assert len(str_funcs) == 0, "Unregistered str should return empty list"
    
    # Additional test: register a function and then query to ensure mock_conversion_func works
    # Create a test class for registration
    class TestRegistrationClass:
        pass
    
    # Register mock_conversion_func for TestRegistrationClass
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=TestRegistrationClass,
        conversion_func=mock_conversion_func,
        priority=100
    )
    
    # Query for TestRegistrationClass
    registered_funcs = tensor_conversion_registry.get(TestRegistrationClass)
    assert len(registered_funcs) > 0, "Should have registered function"
    
    # Check that mock_conversion_func is in the results
    found_mock_func = False
    for base_type, func in registered_funcs:
        if func is mock_conversion_func:
            found_mock_func = True
            break
    assert found_mock_func, "mock_conversion_func should be in query results"
    
    # Test calling the mock function
    test_value = TestRegistrationClass()
    result = mock_conversion_func(test_value, dtype=None, name="test", as_ref=False)
    assert result is not None, "mock_conversion_func should return a mock object"
    
    # Test numeric type default conversion functions
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
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
def test_priority_order_verification(custom_class, func_low, func_high):
    """Test that conversion functions are returned in priority order.
    
    TC-04: 优先级排序验证
    Weak asserts: returns_list, correct_order, contains_all
    """
    # Register functions with different priorities
    # func_high has priority 50 (lower number = higher priority)
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=func_high,
        priority=50
    )
    
    # func_low has priority 200 (higher number = lower priority)
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=custom_class,
        conversion_func=func_low,
        priority=200
    )
    
    # Query for conversion functions
    conversion_funcs = tensor_conversion_registry.get(custom_class)
    
    # Weak assertion 1: returns_list
    assert isinstance(conversion_funcs, list), "Should return a list"
    
    # Weak assertion 2: correct_order
    # Functions should be returned in increasing priority order (lower number first)
    assert len(conversion_funcs) >= 2, "Should have at least two registered functions"
    
    # Check order: func_high (priority 50) should come before func_low (priority 200)
    func_high_found = False
    func_low_found = False
    func_high_index = -1
    func_low_index = -1
    
    for i, (base_type, func) in enumerate(conversion_funcs):
        if func is func_high:
            func_high_found = True
            func_high_index = i
        if func is func_low:
            func_low_found = True
            func_low_index = i
    
    # Weak assertion 3: contains_all
    assert func_high_found, "func_high should be in results"
    assert func_low_found, "func_low should be in results"
    
    # Verify order: higher priority (lower number) should come first
    if func_high_found and func_low_found:
        assert func_high_index < func_low_index, \
            f"Higher priority function (priority 50) should come before lower priority (200). " \
            f"Found func_high at index {func_high_index}, func_low at {func_low_index}"
    
    # Test with three functions (from param_extensions)
    # This tests more complex ordering
    class AnotherCustomClass:
        pass
    
    # Register three functions with different priorities
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=AnotherCustomClass,
        conversion_func=func1,
        priority=30
    )
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=AnotherCustomClass,
        conversion_func=func2,
        priority=10  # Highest priority (lowest number)
    )
    
    tensor_conversion_registry.register_tensor_conversion_function(
        base_type=AnotherCustomClass,
        conversion_func=func3,
        priority=20  # Middle priority
    )
    
    # Get functions for AnotherCustomClass
    another_funcs = tensor_conversion_registry.get(AnotherCustomClass)
    assert len(another_funcs) == 3, "Should have three registered functions"
    
    # Verify order: priority 10, then 20, then 30
    func_indices = {}
    for i, (base_type, func) in enumerate(another_funcs):
        if func is func1:
            func_indices['func1'] = i
        elif func is func2:
            func_indices['func2'] = i
        elif func is func3:
            func_indices['func3'] = i
    
    # Check all functions found
    assert 'func1' in func_indices, "func1 should be in results"
    assert 'func2' in func_indices, "func2 should be in results"
    assert 'func3' in func_indices, "func3 should be in results"
    
    # Verify order: func2 (priority 10) < func3 (priority 20) < func1 (priority 30)
    assert func_indices['func2'] < func_indices['func3'], \
        "func2 (priority 10) should come before func3 (priority 20)"
    assert func_indices['func3'] < func_indices['func1'], \
        "func3 (priority 20) should come before func1 (priority 30)"
    
    # Test that functions with same priority maintain registration order
    # (This is mentioned in the docstring: "order of priority, followed by order of registration")
    same_priority_func1 = lambda v, d=None, n=None, a=False: mock.MagicMock(name="same1")
    same_priority_func2 = lambda v, d=None, n=None, a=False: mock.MagicMock(name="same2")
    
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
    
    # Get updated functions
    updated_funcs = tensor_conversion_registry.get(custom_class)
    
    # Find the new functions
    same1_index = -1
    same2_index = -1
    for i, (base_type, func) in enumerate(updated_funcs):
        try:
            if func.__name__ == '<lambda>' and 'same1' in str(func):
                same1_index = i
            elif func.__name__ == '<lambda>' and 'same2' in str(func):
                same2_index = i
        except (AttributeError, TypeError):
            pass
    
    # If both found, same_priority_func1 should come before same_priority_func2
    # (registered first, so should appear first among same priority)
    if same1_index != -1 and same2_index != -1:
        assert same1_index < same2_index, \
            "Functions with same priority should maintain registration order"
    
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
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_07 START ====
# Placeholder for CASE_07: (DEFERRED)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# Placeholder for CASE_08: (DEFERRED)
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:FOOTER START ====
# Additional helper functions and cleanup for G2
def test_get_function_exists():
    """Test that get function exists and is callable."""
    assert hasattr(tensor_conversion_registry, 'get')
    assert callable(tensor_conversion_registry.get)
    
def test_get_signature():
    """Test that get function has correct signature."""
    import inspect
    sig = inspect.signature(tensor_conversion_registry.get)
    params = list(sig.parameters.keys())
    assert params == ['query'], "get() should have single parameter 'query'"
# ==== BLOCK:FOOTER END ====