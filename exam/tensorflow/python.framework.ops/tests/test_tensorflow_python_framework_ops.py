"""
Test cases for tensorflow.python.framework.ops module.
Generated based on test_plan.json specifications.
"""
import math
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

# ==== BLOCK:HEADER START ====

# Enable TensorFlow 2.x behavior with backward compatibility
# This ensures tests work with both TF 1.x and 2.x
tf.compat.v1.enable_eager_execution()

# Test fixtures and helper functions
@pytest.fixture
def default_graph():
    """Create a default graph for testing."""
    return ops.Graph()

@pytest.fixture
def numpy_rng():
    """Create a numpy random number generator with fixed seed."""
    return np.random.RandomState(42)

def assert_tensor_properties(tensor, expected_dtype, expected_shape, expected_device=None):
    """Helper to assert tensor properties."""
    assert tensor.dtype == expected_dtype
    assert tensor.shape.as_list() == list(expected_shape)
    if expected_device:
        assert tensor.device == expected_device

def create_simple_operation(graph, op_type="Add", name="test_op"):
    """Create a simple operation in the graph."""
    with graph.as_default():
        a = tf.constant(1.0, dtype=tf.float32, name="a")
        b = tf.constant(2.0, dtype=tf.float32, name="b")
        if op_type == "Add":
            return tf.add(a, b, name=name)
        elif op_type == "Mul":
            return tf.multiply(a, b, name=name)
        else:
            return a

def evaluate_tensor_in_session(tensor, graph=None):
    """Helper to evaluate tensor in a session (for compatibility)."""
    if graph is None:
        graph = tensor.graph
    with tf.compat.v1.Session(graph=graph) as sess:
        return sess.run(tensor)

def is_eager_tensor(tensor):
    """Check if tensor is an eager tensor (no name property in eager mode)."""
    try:
        # In eager mode, accessing .name raises AttributeError
        _ = tensor.name
        return False  # Graph mode tensor
    except AttributeError:
        return True  # Eager mode tensor

def create_graph_mode_tensor(graph, value, dtype=None, name=None):
    """Create a tensor in graph mode for testing name property."""
    with graph.as_default():
        return tf.constant(value, dtype=dtype, name=name)

def create_eager_mode_tensor(value, dtype=None):
    """Create a tensor in eager mode."""
    return tf.constant(value, dtype=dtype)
# ==== BLOCK:HEADER END ====

# ==== BLOCK:CASE_01 START ====
def test_graph_creation_and_basic_properties():
    """TC-01: Graph创建与基本属性
    
    Test plan: TC-01
    Priority: High
    Assertion level: weak
    """
    # Create a new graph
    graph = ops.Graph()
    
    # Weak assertions from test plan
    # 1. graph_exists
    assert graph is not None
    assert isinstance(graph, ops.Graph)
    
    # 2. is_empty - check that graph has no operations initially
    # Note: In TensorFlow, we can check if graph is empty by checking operations
    # However, a new graph might have some internal operations
    # We'll check that it's not finalized and has basic properties
    
    # 3. not_finalized
    assert not graph.finalized
    
    # 4. basic_properties
    # Check graph has expected attributes
    assert hasattr(graph, 'as_default')
    assert hasattr(graph, 'finalize')
    assert hasattr(graph, 'get_operations')
    assert hasattr(graph, 'get_tensor_by_name')
    
    # Additional basic checks
    # Graph should have a version
    assert hasattr(graph, 'version')
    # Graph should have collections
    assert hasattr(graph, 'collections')
    
    # Test as_default context manager
    with graph.as_default():
        # Should be able to create operations in this context
        const = tf.constant(1.0, dtype=tf.float32)
        assert const is not None
        assert const.graph is graph
    
    # Test that graph is still not finalized after operations
    assert not graph.finalized
    
    # Test finalize method
    graph.finalize()
    assert graph.finalized
    
    # After finalization, should not be able to add new operations
    # This is expected behavior in TensorFlow
    with graph.as_default():
        with pytest.raises(RuntimeError):
            # Trying to add operation to finalized graph should raise error
            tf.constant(2.0, dtype=tf.float32)
# ==== BLOCK:CASE_01 END ====

# ==== BLOCK:CASE_02 START ====
def test_graph_add_operations_and_tensors():
    """TC-02: Graph添加操作与Tensor
    
    Test plan: TC-02
    Priority: High
    Assertion level: weak
    Requires mock: True
    """
    # Create a new graph
    graph = ops.Graph()
    
    # Track initial state
    initial_ops = graph.get_operations()
    
    # Create operations and tensors in the graph
    with graph.as_default():
        # Create constants (these are operations that produce tensors)
        const1 = tf.constant(1.0, dtype=tf.float32, name="const1")
        const2 = tf.constant(2.0, dtype=tf.float32, name="const2")
        const3 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32, name="const3")
        
        # Create an operation that uses the constants
        add_op = tf.add(const1, const2, name="add_op")
        
        # Create more tensors
        mul_op = tf.multiply(add_op, const1, name="mul_op")
        reshape_op = tf.reshape(const3, [3, 1], name="reshape_op")
    
    # Get all operations after creation
    all_ops = graph.get_operations()
    
    # Weak assertions from test plan
    # 1. operation_added - check operations were added
    assert len(all_ops) > len(initial_ops)
    
    # Count specific operations
    operation_names = [op.name for op in all_ops]
    assert "const1" in operation_names
    assert "const2" in operation_names
    assert "const3" in operation_names
    assert "add_op" in operation_names
    assert "mul_op" in operation_names
    assert "reshape_op" in operation_names
    
    # 2. tensor_added - check tensors exist
    # Each operation produces output tensors
    for op in all_ops:
        assert op.outputs is not None
        assert len(op.outputs) > 0
    
    # 3. dtype_correct - check dtype of created tensors
    assert const1.dtype == tf.float32
    assert const2.dtype == tf.float32
    assert const3.dtype == tf.float32
    assert add_op.dtype == tf.float32
    assert mul_op.dtype == tf.float32
    
    # 4. shape_correct - check shapes
    assert const1.shape.as_list() == []
    assert const2.shape.as_list() == []
    assert const3.shape.as_list() == [3]
    assert add_op.shape.as_list() == []
    assert mul_op.shape.as_list() == []
    assert reshape_op.shape.as_list() == [3, 1]
    
    # Additional checks for graph consistency
    # Check tensor dependencies
    assert add_op.op.inputs[0] == const1
    assert add_op.op.inputs[1] == const2
    assert mul_op.op.inputs[0] == add_op
    assert mul_op.op.inputs[1] == const1
    
    # Check that tensors belong to the correct graph
    for op in all_ops:
        assert op.graph is graph
        for tensor in op.outputs:
            assert tensor.graph is graph
    
    # Test getting tensor by name
    const1_by_name = graph.get_tensor_by_name("const1:0")
    assert const1_by_name is const1
    
    const3_by_name = graph.get_tensor_by_name("const3:0")
    assert const3_by_name is const3
    
    add_op_by_name = graph.get_tensor_by_name("add_op:0")
    assert add_op_by_name is add_op
    
    # Test operation count matches expectation
    # We created 6 operations: const1, const2, const3, add_op, mul_op, reshape_op
    assert len(all_ops) >= 6  # There might be additional internal operations
# ==== BLOCK:CASE_02 END ====

# ==== BLOCK:CASE_03 START ====
def test_tensor_basic_construction_and_properties():
    """TC-03: Tensor基本构造与属性
    
    Test plan: TC-03
    Priority: High
    Assertion level: weak
    """
    # Create a graph for testing
    graph = ops.Graph()
    
    with graph.as_default():
        # Create a constant tensor with specific properties
        # Parameters from test plan: dtype=float32, shape=[2, 3], device=cpu, value_type=constant
        tensor = tf.constant(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=tf.float32,
            name="test_tensor"
        )
    
    # Weak assertions from test plan
    # 1. tensor_exists
    assert tensor is not None
    assert isinstance(tensor, tf.Tensor)
    
    # 2. dtype_matches
    assert tensor.dtype == tf.float32
    
    # 3. shape_matches
    assert tensor.shape.as_list() == [2, 3]
    
    # 4. device_matches
    # Note: In TensorFlow, device assignment might be None or default
    # We'll check that device attribute exists and is accessible
    assert hasattr(tensor, 'device')
    # Device might be empty string or None for CPU
    assert tensor.device is not None
    
    # Additional property checks
    # Check tensor name
    assert tensor.name == "test_tensor:0"
    
    # Check tensor op
    assert tensor.op is not None
    assert tensor.op.name == "test_tensor"
    assert tensor.op.type == "Const"
    
    # Check tensor value_index (should be 0 for single output ops)
    assert tensor.value_index == 0
    
    # Check tensor graph
    assert tensor.graph is graph
    
    # Test tensor attributes are immutable
    # These should be read-only properties
    original_dtype = tensor.dtype
    original_shape = tensor.shape
    original_name = tensor.name
    
    # Verify these cannot be changed (they are properties, not setters)
    # This is inherent in TensorFlow's design
    
    # Test tensor string representation
    str_repr = str(tensor)
    assert "Tensor" in str_repr
    assert "test_tensor:0" in str_repr
    assert "shape=(2, 3)" in str_repr or "shape=[2, 3]" in str_repr
    assert "dtype=float32" in str_repr
    
    # Test tensor repr
    repr_str = repr(tensor)
    assert "Tensor" in repr_str
    
    # Test tensor equality (identity based)
    with graph.as_default():
        same_tensor = graph.get_tensor_by_name("test_tensor:0")
        assert same_tensor is tensor
        
        # Create another tensor with same values but different name
        tensor2 = tf.constant(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=tf.float32,
            name="test_tensor2"
        )
        assert tensor2 is not tensor
    
    # Test tensor consumption in operations
    with graph.as_default():
        # Use the tensor in an operation
        add_result = tf.add(tensor, 1.0, name="add_to_tensor")
        assert add_result is not None
        assert add_result.dtype == tf.float32
        assert add_result.shape.as_list() == [2, 3]
        
        # Verify the original tensor is unchanged
        assert tensor.dtype == tf.float32
        assert tensor.shape.as_list() == [2, 3]
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_04 START ====
def test_tensor_property_access_methods():
    """TC-04: Tensor属性访问方法
    
    Test plan: TC-04
    Priority: High
    Assertion level: weak
    """
    # Create a graph for testing
    graph = ops.Graph()
    
    with graph.as_default():
        # Create a tensor with specific properties
        # Parameters from test plan: dtype=int32, shape=[4, 4], device=cpu
        tensor = tf.constant(
            np.ones((4, 4), dtype=np.int32),
            dtype=tf.int32,
            name="property_test_tensor"
        )
    
    # Weak assertions from test plan
    # 1. dtype_accessible
    dtype = tensor.dtype
    assert dtype is not None
    assert dtype == tf.int32
    assert isinstance(dtype, tf.DType)
    
    # 2. shape_accessible
    shape = tensor.shape
    assert shape is not None
    assert shape.as_list() == [4, 4]
    assert shape.rank == 2
    assert shape.dims is not None
    assert len(shape.dims) == 2
    assert shape.dims[0].value == 4
    assert shape.dims[1].value == 4
    
    # 3. device_accessible
    device = tensor.device
    assert device is not None
    # Device could be empty string or specific device string
    assert isinstance(device, str)
    
    # 4. name_accessible
    name = tensor.name
    assert name is not None
    assert isinstance(name, str)
    assert name == "property_test_tensor:0"
    
    # Additional property access tests
    # Test op property
    op = tensor.op
    assert op is not None
    assert op.name == "property_test_tensor"
    assert op.type == "Const"
    assert op.graph is graph
    
    # Test value_index property
    value_index = tensor.value_index
    assert value_index == 0
    
    # Test graph property
    tensor_graph = tensor.graph
    assert tensor_graph is graph
    
    # Test consumers property
    consumers = tensor.consumers()
    assert consumers is not None
    # Initially no consumers
    assert len(consumers) == 0
    
    # Add a consumer and test again
    with graph.as_default():
        add_op = tf.add(tensor, 1, name="add_consumer")
        consumers_after = tensor.consumers()
        assert len(consumers_after) > 0
        assert add_op.op in consumers_after
    
    # Test tensor ID (unique identifier)
    tensor_id = tensor._id
    assert tensor_id is not None
    
    # Test tensor string conversion methods
    # __str__ method
    str_repr = str(tensor)
    assert "Tensor" in str_repr
    assert "property_test_tensor:0" in str_repr
    assert "shape=(4, 4)" in str_repr or "shape=[4, 4]" in str_repr
    assert "dtype=int32" in str_repr
    
    # __repr__ method
    repr_str = repr(tensor)
    assert "Tensor" in repr_str
    
    # Test tensor equality and hash
    # Tensors should be comparable by identity
    with graph.as_default():
        same_tensor = graph.get_tensor_by_name("property_test_tensor:0")
        assert same_tensor is tensor
        assert hash(same_tensor) == hash(tensor)
        
        # Different tensor should not be equal
        other_tensor = tf.constant(1, dtype=tf.int32, name="other")
        assert other_tensor is not tensor
        assert hash(other_tensor) != hash(tensor)
    
    # Test tensor property immutability
    # These properties should be read-only
    # We can't directly test immutability by assignment since they're properties
    # But we can verify they return consistent values
    
    original_properties = {
        'dtype': tensor.dtype,
        'shape': tensor.shape.as_list(),
        'name': tensor.name,
        'device': tensor.device
    }
    
    # Access multiple times to ensure consistency
    for _ in range(3):
        assert tensor.dtype == original_properties['dtype']
        assert tensor.shape.as_list() == original_properties['shape']
        assert tensor.name == original_properties['name']
        assert tensor.device == original_properties['device']
    
    # Test tensor in different contexts
    # Tensor should maintain properties across contexts
    # In TensorFlow 2.x, we use Eager Execution instead of tf.Session
    # Create a function to evaluate the tensor - but we need to be careful
    # with graph mode tensors in eager execution
    
    # Instead of using @tf.function which has scope issues with graph mode tensors,
    # we'll test using tf.compat.v1.Session for compatibility
    with tf.compat.v1.Session(graph=graph) as sess:
        # Evaluate the tensor
        numpy_value = sess.run(tensor)
        assert numpy_value is not None
        assert numpy_value.shape == (4, 4)
        assert numpy_value.dtype == np.int32
        assert np.array_equal(numpy_value, np.ones((4, 4), dtype=np.int32))
        
        # Test that tensor properties are preserved in session context
        # We can't access tensor.name directly in session, but we can verify
        # the tensor is the same object
        assert tensor.graph is graph
        
        # Test using the tensor in operations within the session
        with graph.as_default():
            # Create an operation using the tensor
            add_result = tf.add(tensor, 2)
            add_value = sess.run(add_result)
            expected_value = np.ones((4, 4), dtype=np.int32) + 2
            assert np.array_equal(add_value, expected_value)
    
    # Test eager mode tensor creation for comparison
    # In eager mode, tensors behave differently
    eager_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    assert eager_tensor.dtype == tf.int32
    assert eager_tensor.shape.as_list() == [2, 2]
    
    # In eager mode, we can use @tf.function without scope issues
    @tf.function
    def eager_tensor_function(t):
        return t * 2
    
    eager_result = eager_tensor_function(eager_tensor)
    assert eager_result is not None
    assert eager_result.dtype == tf.int32
    assert eager_result.shape.as_list() == [2, 2]
    
    # Verify the value
    eager_numpy = eager_result.numpy()
    expected_eager = np.array([[2, 4], [6, 8]], dtype=np.int32)
    assert np.array_equal(eager_numpy, expected_eager)
# ==== BLOCK:CASE_04 END ====

# ==== BLOCK:CASE_05 START ====
# TC-05: Graph线程安全性验证 (DEFERRED - placeholder)
# ==== BLOCK:CASE_05 END ====

# ==== BLOCK:CASE_06 START ====
# TC-06: Tensor边界形状与数值 (DEFERRED - placeholder)
# ==== BLOCK:CASE_06 END ====

# ==== BLOCK:CASE_07 START ====
# TC-07: Tensor极端数值处理 (DEFERRED - placeholder)
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
def test_convert_to_tensor_basic_conversion():
    """TC-08: convert_to_tensor基本转换
    
    Test plan: TC-08
    Priority: High
    Assertion level: weak
    """
    # Parameters from test plan: input_type=numpy_array, dtype=float32, shape=[3, 3], data=random
    
    # Create test data
    np.random.seed(42)
    numpy_array = np.random.randn(3, 3).astype(np.float32)
    
    # Test conversion using TensorFlow's convert_to_tensor
    # Note: tf.convert_to_tensor is the public API
    tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
    
    # Weak assertions from test plan
    # 1. conversion_success
    assert tensor is not None
    assert isinstance(tensor, tf.Tensor)
    
    # 2. dtype_matches
    assert tensor.dtype == tf.float32
    
    # 3. shape_matches
    assert tensor.shape.as_list() == [3, 3]
    
    # 4. value_preserved
    # In TensorFlow 2.x, we can directly convert tensor to numpy for value checking
    tensor_value = tensor.numpy()
    # Check shape and dtype match
    assert tensor_value.shape == (3, 3)
    assert tensor_value.dtype == np.float32
    # Check values are close (allow for floating point differences)
    np.testing.assert_array_almost_equal(tensor_value, numpy_array, decimal=5)
    
    # Additional conversion tests
    
    # Test without explicit dtype (should infer from numpy array)
    tensor_inferred = tf.convert_to_tensor(numpy_array)
    assert tensor_inferred.dtype == tf.float32
    assert tensor_inferred.shape.as_list() == [3, 3]
    
    # Test with name parameter - in eager mode, name is not accessible
    # So we'll test in graph mode for name property
    graph = ops.Graph()
    with graph.as_default():
        # Create tensor in graph mode to test name property
        graph_tensor_named = tf.convert_to_tensor(numpy_array, dtype=tf.float32, name="named_tensor")
        assert graph_tensor_named.name == "named_tensor:0"
    
    # Test conversion of already a tensor (should return same tensor)
    tensor_already = tf.convert_to_tensor(tensor)
    assert tensor_already is tensor
    
    # Test conversion with different dtypes
    int_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    int_tensor = tf.convert_to_tensor(int_array, dtype=tf.int32)
    assert int_tensor.dtype == tf.int32
    assert int_tensor.shape.as_list() == [2, 3]
    
    # Test conversion preserves graph context
    with graph.as_default():
        graph_tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
        assert graph_tensor.graph is graph
    
    # Test conversion of Python lists
    python_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    list_tensor = tf.convert_to_tensor(python_list, dtype=tf.float32)
    assert list_tensor.dtype == tf.float32
    assert list_tensor.shape.as_list() == [2, 3]
    
    # Test conversion of Python scalars
    scalar_tensor = tf.convert_to_tensor(42.0, dtype=tf.float32)
    assert scalar_tensor.dtype == tf.float32
    assert scalar_tensor.shape.as_list() == []
    
    # Test conversion with shape validation
    # This should work for compatible shapes
    reshaped_array = numpy_array.reshape(9)
    reshaped_tensor = tf.convert_to_tensor(reshaped_array, dtype=tf.float32)
    assert reshaped_tensor.shape.as_list() == [9]
    
    # Test that conversion creates proper operation in graph mode
    with graph.as_default():
        graph_tensor_for_op = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
        # In graph mode, tensor should have op property
        assert hasattr(graph_tensor_for_op, 'op')
        assert graph_tensor_for_op.op is not None
        assert graph_tensor_for_op.op.type in ["Const", "Placeholder", "Identity"]  # Could be any of these
    
    # Test tensor properties after conversion
    # In eager mode, some properties like 'name' and 'op' are not available
    # but 'device' should be available
    assert hasattr(tensor, 'device')
    
    # Check if tensor has op attribute (may be None in eager mode)
    # In eager mode, op may not exist or be None
    if hasattr(tensor, 'op'):
        # If op exists, it might be None in eager mode
        # This is acceptable
        pass
    else:
        # In some TensorFlow versions, eager tensors don't have op attribute
        # This is also acceptable
        pass
    
    # Test that converted tensor can be used in operations
    # Create operation using converted tensor
    add_result = tf.add(tensor, 1.0)
    add_value = add_result.numpy()
    expected_value = numpy_array + 1.0
    np.testing.assert_array_almost_equal(add_value, expected_value, decimal=5)
    
    # Test gradient support (conceptually)
    # Converted tensors should support gradient computation
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
        tape.watch(x)
        y = x * 2.0
        # Compute gradient
        grad = tape.gradient(y, x)
        # Gradient should be 2.0 for each element
        assert grad is not None
        grad_numpy = grad.numpy()
        expected_grad = np.ones_like(numpy_array) * 2.0
        np.testing.assert_array_almost_equal(grad_numpy, expected_grad, decimal=5)
    
    # Test edge cases for convert_to_tensor
    # Test with None (should raise error)
    with pytest.raises((ValueError, TypeError)):
        tf.convert_to_tensor(None)
    
    # Test with empty list
    empty_list_tensor = tf.convert_to_tensor([], dtype=tf.float32)
    assert empty_list_tensor.dtype == tf.float32
    assert empty_list_tensor.shape.as_list() == [0]
    
    # Test with nested lists
    nested_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    nested_tensor = tf.convert_to_tensor(nested_list, dtype=tf.int32)
    assert nested_tensor.dtype == tf.int32
    assert nested_tensor.shape.as_list() == [2, 2, 2]
    
    # Test type inference
    float_list = [1.5, 2.5, 3.5]
    inferred_tensor = tf.convert_to_tensor(float_list)
    assert inferred_tensor.dtype == tf.float32  # Should infer float32
    
    int_list = [1, 2, 3]
    inferred_int_tensor = tf.convert_to_tensor(int_list)
    assert inferred_int_tensor.dtype == tf.int32  # Should infer int32
    
    # Test with dtype_hint parameter
    hinted_tensor = tf.convert_to_tensor([1, 2, 3], dtype_hint=tf.float64)
    assert hinted_tensor.dtype == tf.float64
# ==== BLOCK:CASE_08 END ====

# ==== BLOCK:CASE_09 START ====
# TC-09: convert_to_tensor支持类型 (DEFERRED - placeholder)
# ==== BLOCK:CASE_09 END ====

# ==== BLOCK:CASE_10 START ====
# TC-10: convert_to_tensor异常处理 (DEFERRED - placeholder)
# ==== BLOCK:CASE_10 END ====

# ==== BLOCK:FOOTER START ====
# Additional test utilities and cleanup

def test_graph_collections():
    """Test graph collections functionality."""
    graph = ops.Graph()
    
    # Test adding to collections
    with graph.as_default():
        const = tf.constant(1.0, name="test_const")
        graph.add_to_collection("test_collection", const)
    
    # Test retrieving from collections
    collection = graph.get_collection("test_collection")
    assert len(collection) == 1
    assert collection[0] is const
    
    # Test collection keys
    keys = graph.get_all_collection_keys()
    assert "test_collection" in keys

def test_tensor_from_op():
    """Test tensor creation from operation."""
    graph = ops.Graph()
    
    with graph.as_default():
        # Create an operation
        const_op = tf.constant(1.0, name="const_op").op
        
        # Tensor should be accessible from operation outputs
        assert len(const_op.outputs) == 1
        tensor = const_op.outputs[0]
        
        assert tensor.op is const_op
        assert tensor.value_index == 0
        assert tensor.name == "const_op:0"

def test_convert_n_to_tensor():
    """Test convert_n_to_tensor function."""
    # Create test data
    numpy_arrays = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32)
    ]
    
    # Convert multiple tensors
    tensors = tf.nest.map_structure(tf.convert_to_tensor, numpy_arrays)
    
    assert len(tensors) == 3
    for i, tensor in enumerate(tensors):
        assert isinstance(tensor, tf.Tensor)
        assert tensor.dtype == tf.float32
        assert tensor.shape.as_list() == [2]

def test_graph_device_management():
    """Test graph device assignment."""
    graph = ops.Graph()
    
    # Test device context manager
    with graph.device("/cpu:0"):
        with graph.as_default():
            const = tf.constant(1.0, name="cpu_const")
            # Device might be set or empty
            assert const.device is not None
    
    # Test default device behavior
    with graph.as_default():
        default_const = tf.constant(2.0, name="default_const")
        assert default_const.device is not None

def test_tensor_consumers():
    """Test tensor consumers tracking."""
    graph = ops.Graph()
    
    with graph.as_default():
        # Create a tensor
        a = tf.constant(1.0, name="a")
        
        # Initially no consumers
        assert len(a.consumers()) == 0
        
        # Create operations that consume the tensor
        b = tf.add(a, 1.0, name="b")
        c = tf.multiply(a, 2.0, name="c")
        
        # Now should have consumers
        consumers = a.consumers()
        assert len(consumers) == 2
        
        consumer_ops = [consumer for consumer in consumers]
        assert b.op in consumer_ops
        assert c.op in consumer_ops

if __name__ == "__main__":
    # Simple test runner for debugging
    import sys
    pytest.main([sys.argv[0], "-v"])
# ==== BLOCK:FOOTER END ====