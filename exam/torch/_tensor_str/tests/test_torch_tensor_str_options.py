import torch
import pytest
import math
from unittest.mock import patch, MagicMock
from torch._tensor_str import set_printoptions, PRINT_OPTS

import torch
import pytest
import math
from unittest.mock import patch, MagicMock
from torch._tensor_str import set_printoptions, PRINT_OPTS


class TestTensorStrOptions:
    """Test cases for print options configuration."""
    
    def setup_method(self):
        """Save original print options before each test."""
        self.original_opts = {
            'precision': PRINT_OPTS.precision,
            'threshold': PRINT_OPTS.threshold,
            'edgeitems': PRINT_OPTS.edgeitems,
            'linewidth': PRINT_OPTS.linewidth,
            'sci_mode': PRINT_OPTS.sci_mode
        }
    
    def teardown_method(self):
        """Restore original print options after each test."""
        set_printoptions(
            precision=self.original_opts['precision'],
            threshold=self.original_opts['threshold'],
            edgeitems=self.original_opts['edgeitems'],
            linewidth=self.original_opts['linewidth'],
            sci_mode=self.original_opts['sci_mode']
        )

# ==== BLOCK:CASE_03 START ====
    @pytest.mark.parametrize("precision,threshold,edgeitems,linewidth,sci_mode", [
        (2, 5, 2, 80, False),
        (6, 10, 3, 120, True),  # Parameter extension from test plan
    ])
    def test_global_print_options_setting(self, precision, threshold, edgeitems, linewidth, sci_mode):
        """TC-03: Global print options setting and verification."""
        # Save original state
        original_precision = PRINT_OPTS.precision
        original_threshold = PRINT_OPTS.threshold
        
        # Set new options
        set_printoptions(
            precision=precision,
            threshold=threshold,
            edgeitems=edgeitems,
            linewidth=linewidth,
            sci_mode=sci_mode
        )
        
        # Weak assertions
        # 1. Options set successfully
        assert PRINT_OPTS.precision == precision, \
            f"Precision should be {precision}, got {PRINT_OPTS.precision}"
        assert PRINT_OPTS.threshold == threshold, \
            f"Threshold should be {threshold}, got {PRINT_OPTS.threshold}"
        assert PRINT_OPTS.edgeitems == edgeitems, \
            f"Edgeitems should be {edgeitems}, got {PRINT_OPTS.edgeitems}"
        assert PRINT_OPTS.linewidth == linewidth, \
            f"Linewidth should be {linewidth}, got {PRINT_OPTS.linewidth}"
        assert PRINT_OPTS.sci_mode == sci_mode, \
            f"Sci_mode should be {sci_mode}, got {PRINT_OPTS.sci_mode}"
        
        # 2. Global state changed
        # Create a tensor to test the effect
        tensor = torch.arange(10, dtype=torch.float32)
        tensor_str = str(tensor)
        
        # 3. Affects subsequent formatting
        # With threshold=5, tensor of size 10 should be truncated
        if threshold < 10:
            assert "..." in tensor_str, \
                f"Tensor with {len(tensor)} elements should be truncated with threshold={threshold}"
        
        # Check precision effect
        if precision < original_precision:
            # Find floating point numbers
            import re
            floats = re.findall(r'[-+]?\d+\.\d+', tensor_str)
            if floats:
                sample_float = floats[0]
                decimal_part = sample_float.split('.')[1]
                assert len(decimal_part) <= precision, \
                    f"Decimal places ({len(decimal_part)}) should not exceed precision ({precision})"
        
        # Test reset functionality
        set_printoptions(profile='default')
        assert PRINT_OPTS.precision == 4, "Default precision should be 4"
        assert PRINT_OPTS.threshold == 1000, "Default threshold should be 1000"
# ==== BLOCK:CASE_03 END ====

# ==== BLOCK:CASE_07 START ====
# Placeholder for CASE_07: Deferred test case
# ==== BLOCK:CASE_07 END ====

# ==== BLOCK:CASE_08 START ====
# Placeholder for CASE_08: Deferred test case
# ==== BLOCK:CASE_08 END ====

if __name__ == "__main__":
    pytest.main([__file__, "-v"])