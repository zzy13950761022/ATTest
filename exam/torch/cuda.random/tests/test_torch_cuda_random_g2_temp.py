import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, call
from typing import List

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Skip tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), 
                                reason="CUDA is not available")

class TestTorchCudaRandomG2:
    """Test class for torch.cuda.random multi-device state management functions.
    
    This class tests the following functions:
    - get_rng_state_all: Get RNG states for all devices
    - set_rng_state_all: Set RNG states for all devices
    - manual_seed_all: Set seed for all GPUs
    - seed_all: Set random seed for all GPUs
    
    Test group: G2 (Multi-device state management)
    """
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Save original CUDA RNG state to restore after test
        if torch.cuda.is_available():
            self.original_states = torch.cuda.get_rng_state_all()
        yield
        # Restore original CUDA RNG state
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.original_states)
    
    @pytest.fixture
    def mock_device_count(self, count: int = 2):
        """Mock device count for tests.
        
        Args:
            count: Number of devices to mock
        """
        with patch('torch.cuda.device_count', return_value=count):
            yield
    
    @pytest.fixture
    def mock_cuda_generators(self):
        """Mock CUDA default generators for testing."""
        mock_generators = []
        for i in range(2):  # Assume 2 devices
            mock_gen = MagicMock()
            mock_gen.get_state.return_value = torch.ByteTensor(100).random_(0, 256)
            mock_generators.append(mock_gen)
        
        with patch('torch.cuda.default_generators', mock_generators):
            yield mock_generators

    # ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize("device_count,state_type,seed,test_scenario", [
        (2, "valid_byte_tensor_list", 42, "multi_device_basic"),
    ])
    def test_multi_device_state_batch_management(self, device_count, state_type, seed, test_scenario):
        """Test multi-device state batch management operations.
        
        TC-03: 多设备状态批量管理
        Priority: High
        Assertion level: weak
        
        Test scenarios:
        - multi_device_basic: Basic multi-device state management
        
        Weak assertions:
        - list_length_matches_device_count: List length should match device count
        - each_state_is_byte_tensor: Each state should be ByteTensor type
        - no_exception: No exceptions should be raised
        - all_devices_processed: All devices should be processed
        """
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")
        
        # Mock device count to match parameter
        with patch('torch.cuda.device_count', return_value=device_count):
            # Test 1: Get states for all devices
            all_states = torch.cuda.get_rng_state_all()
            
            # Weak assertion: list_length_matches_device_count
            assert isinstance(all_states, list), "get_rng_state_all should return a list"
            assert len(all_states) == device_count, \
                f"Should return {device_count} states for {device_count} devices, got {len(all_states)}"
            
            # Weak assertion: each_state_is_byte_tensor
            for i, state in enumerate(all_states):
                assert isinstance(state, torch.Tensor), \
                    f"State {i} should be a Tensor"
                assert state.dtype == torch.uint8, \
                    f"State {i} should be ByteTensor, got {state.dtype}"
                assert state.dim() == 1, \
                    f"State {i} should be 1-dimensional"
            
            # Test 2: Create test states for all devices
            test_states = []
            for i in range(device_count):
                # Create unique state for each device
                state_size = 100 + i * 10  # Vary size slightly
                test_state = torch.ByteTensor(state_size).random_(0, 256)
                test_states.append(test_state)
            
            # Test 3: Set states for all devices
            torch.cuda.set_rng_state_all(test_states)
            
            # Test 4: Verify states were set by getting them back
            retrieved_states = torch.cuda.get_rng_state_all()
            
            # Weak assertion: all_devices_processed
            assert len(retrieved_states) == device_count, \
                f"Should retrieve {device_count} states"
            
            # Verify each state was set correctly
            for i, (test_state, retrieved_state) in enumerate(zip(test_states, retrieved_states)):
                assert retrieved_state.shape == test_state.shape, \
                    f"Device {i}: Retrieved shape {retrieved_state.shape} != test shape {test_state.shape}"
                # For ByteTensor, we can compare exact values
                assert torch.equal(retrieved_state, test_state), \
                    f"Device {i}: State values not preserved"
            
            # Test 5: Test manual_seed_all
            torch.cuda.manual_seed_all(seed)
            
            # Verify seed affects all devices by checking random generation
            # Generate random numbers on each device
            random_seqs = []
            for i in range(device_count):
                torch.cuda.set_device(i)
                torch.cuda.manual_seed_all(seed)  # Reset seed for all
                seq = torch.cuda.FloatTensor(10).normal_()
                random_seqs.append(seq)
            
            # Reset seed and generate again
            torch.cuda.manual_seed_all(seed)
            random_seqs2 = []
            for i in range(device_count):
                torch.cuda.set_device(i)
                seq = torch.cuda.FloatTensor(10).normal_()
                random_seqs2.append(seq)
            
            # Verify each device produces identical sequence with same seed
            for i, (seq1, seq2) in enumerate(zip(random_seqs, random_seqs2)):
                assert torch.allclose(seq1, seq2), \
                    f"Device {i}: Random sequences should be identical with same seed"
            
            # Test 6: Test seed_all
            torch.cuda.seed_all()
            # No direct assertion, just verify it doesn't crash
            
            # Weak assertion: no_exception
            # If we reached here without exceptions, test passes
            
            # Test 7: Verify cross-device independence
            # Set different seeds on different devices using manual_seed (not manual_seed_all)
            for i in range(device_count):
                torch.cuda.set_device(i)
                torch.cuda.manual_seed(seed + i)  # Different seed for each device
            
            # Generate sequences
            seqs_different_seeds = []
            for i in range(device_count):
                torch.cuda.set_device(i)
                seq = torch.cuda.FloatTensor(10).normal_()
                seqs_different_seeds.append(seq)
            
            # They should be different from each other (low probability of collision)
            for i in range(device_count):
                for j in range(i + 1, device_count):
                    assert not torch.allclose(seqs_different_seeds[i], seqs_different_seeds[j], rtol=1e-5), \
                        f"Devices {i} and {j} with different seeds should produce different sequences"
    # ==== BLOCK:CASE_03 END ====

    # ==== BLOCK:CASE_04 START ====
    @pytest.mark.parametrize("invalid_device,invalid_state_type,test_scenario", [
        (-1, "float_tensor", "error_handling"),
    ])
    def test_invalid_device_index_exception_handling(self, invalid_device, invalid_state_type, test_scenario):
        """Test invalid device index exception handling.
        
        TC-04: 无效设备索引异常处理
        Priority: High
        Assertion level: weak
        
        Test scenarios:
        - error_handling: Error handling for invalid device indices
        
        Weak assertions:
        - exception_raised: Exception should be raised
        - exception_type_correct: Exception type should be correct
        - error_message_contains_device: Error message should contain device info
        - no_side_effects: No side effects should occur
        """
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")
        
        # Mock device count to control valid range
        with patch('torch.cuda.device_count', return_value=2):
            # Test 1: Invalid device index for get_rng_state
            with pytest.raises(RuntimeError) as exc_info:
                torch.cuda.get_rng_state(device=invalid_device)
            
            # Weak assertion: exception_raised
            assert exc_info.value is not None, "Should raise exception for invalid device"
            
            # Weak assertion: error_message_contains_device
            error_msg = str(exc_info.value).lower()
            assert "device" in error_msg or "index" in error_msg or str(invalid_device) in error_msg, \
                f"Error message should mention device/index, got: {error_msg}"
            
            # Test 2: Invalid device index for set_rng_state
            # First create a valid ByteTensor state
            valid_state = torch.ByteTensor(100).random_(0, 256)
            
            with pytest.raises(RuntimeError) as exc_info2:
                torch.cuda.set_rng_state(valid_state, device=invalid_device)
            
            # Weak assertion: exception_type_correct
            assert isinstance(exc_info2.value, RuntimeError), \
                f"Should raise RuntimeError, got {type(exc_info2.value)}"
            
            # Test 3: Test with large invalid index
            with patch('torch.cuda.device_count', return_value=2):
                with pytest.raises(RuntimeError) as exc_info3:
                    torch.cuda.get_rng_state(device=999)  # Large invalid index
                
                assert exc_info3.value is not None, "Should raise exception for large invalid index"
            
            # Test 4: Verify no side effects on valid devices
            # Get original state of device 0
            original_state_device0 = torch.cuda.get_rng_state(device=0)
            
            # Try invalid operation
            try:
                torch.cuda.get_rng_state(device=invalid_device)
            except RuntimeError:
                pass  # Expected
            
            # Verify device 0 state unchanged
            current_state_device0 = torch.cuda.get_rng_state(device=0)
            assert torch.equal(current_state_device0, original_state_device0), \
                "Valid device state should not be affected by invalid device operation"
            
            # Test 5: Test invalid state type (non-ByteTensor) for set_rng_state
            # This is actually tested in CASE_05, but we can do a basic check here
            float_state = torch.FloatTensor(100).normal_()
            
            with pytest.raises(RuntimeError) as exc_info4:
                torch.cuda.set_rng_state(float_state, device=0)
            
            # Error message should mention type or ByteTensor
            error_msg4 = str(exc_info4.value).lower()
            assert "type" in error_msg4 or "byte" in error_msg4 or "dtype" in error_msg4, \
                f"Error message should mention type/ByteTensor, got: {error_msg4}"
    # ==== BLOCK:CASE_04 END ====

    # ==== BLOCK:CASE_07 START ====
    # Test case: Empty state list handling (DEFERRED)
    # TC-07: 空状态列表处理
    # This is a deferred test case
    def test_empty_state_list_handling(self):
        """Placeholder for empty state list handling test.
        
        TC-07: 空状态列表处理
        Priority: Medium
        Assertion level: weak (when implemented)
        
        This test is deferred and will be implemented in later rounds.
        """
        pytest.skip("DEFERRED: Empty state list handling test")
    # ==== BLOCK:CASE_07 END ====

# ==== BLOCK:FOOTER START ====
# Helper functions and fixtures for G2 tests

def test_g2_module_functions_exist():
    """Test that all G2 module functions exist and are callable."""
    import torch.cuda.random as cuda_random
    
    g2_functions = [
        'get_rng_state_all',
        'set_rng_state_all', 
        'manual_seed_all',
        'seed_all'
    ]
    
    for func_name in g2_functions:
        assert hasattr(cuda_random, func_name), \
            f"torch.cuda.random should have function {func_name} (G2)"
        func = getattr(cuda_random, func_name)
        assert callable(func), f"{func_name} should be callable"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_device_consistency():
    """Test consistency between single-device and multi-device functions."""
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        # Get states using both methods
        state_single = torch.cuda.get_rng_state(device=0)
        all_states = torch.cuda.get_rng_state_all()
        
        # State for device 0 should match
        assert torch.equal(state_single, all_states[0]), \
            "get_rng_state(device=0) should match get_rng_state_all()[0]"
        
        # Test manual_seed vs manual_seed_all
        torch.cuda.manual_seed(42)
        seq_single = torch.cuda.FloatTensor(10).normal_()
        
        torch.cuda.manual_seed_all(42)
        torch.cuda.set_device(0)
        seq_all = torch.cuda.FloatTensor(10).normal_()
        
        # They should be the same when seed is the same
        assert torch.allclose(seq_single, seq_all), \
            "manual_seed(42) and manual_seed_all(42) should produce same sequence on device 0"


class TestTorchCudaRandomCrossGroup:
    """Tests that span both G1 and G2 functionality."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_group_interaction(self):
        """Test interaction between single-device and multi-device functions."""
        if torch.cuda.is_available():
            # Get state using single-device function
            state0_single = torch.cuda.get_rng_state(device=0)
            
            # Get all states
            all_states = torch.cuda.get_rng_state_all()
            
            # They should be consistent
            assert torch.equal(state0_single, all_states[0]), \
                "Single-device and multi-device get should be consistent"
            
            # Modify using single-device function
            new_state = torch.ByteTensor(state0_single.shape).random_(0, 256)
            torch.cuda.set_rng_state(new_state, device=0)
            
            # Verify using multi-device function
            updated_states = torch.cuda.get_rng_state_all()
            assert torch.equal(updated_states[0], new_state), \
                "Single-device set should be visible via multi-device get"
            
            # Modify using multi-device function
            new_states = []
            for i, state in enumerate(updated_states):
                new_state_i = torch.ByteTensor(state.shape).random_(0, 256)
                new_states.append(new_state_i)
            
            torch.cuda.set_rng_state_all(new_states)
            
            # Verify using single-device function
            for i, expected_state in enumerate(new_states):
                actual_state = torch.cuda.get_rng_state(device=i)
                assert torch.equal(actual_state, expected_state), \
                    f"Multi-device set should be visible via single-device get for device {i}"
# ==== BLOCK:FOOTER END ====