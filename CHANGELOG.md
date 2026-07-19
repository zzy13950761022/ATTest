# ATTest Changelog

## v0.4.0 - Coverage Density & Resilience Improvements

Based on B4 baseline analysis (workspace-qwen-3.7-plus_v2) that identified:
- Completion rate: 33/45 (73%), generate_code failure: 27%
- Coverage extraction failure: 49% (22/45), due to generated UT breaking lcov target
- Test case density: 9.3 cases/op (vs 101 for opencode-plain)

### Changes

#### Fix 1: Increased test case density (~9 → ~25+ per operator)
**File:** `src/attest_cli/workflow/ascend_stages.py` - `_build_cases()`

- Generate parametric cases based on `supported_dtypes` from inspection
- Added per-dtype coverage (up to 6 dtypes per layer)
- Added per-format coverage (up to 3 formats for tiling + infershape)
- Added edge cases: scalar, broadcast, high-rank, zero-dim, inplace
- Added: `tiling_scalar_inputs`, `tiling_large_shape`, `tiling_zero_dim`, `tiling_broadcast`
- Added: `infershape_scalar`, `infershape_high_rank`, `infershape_0d`, `infershape_mismatch_rank`, `infershape_null_input`
- Added: `api_null_output`, `api_null_workspace`, `api_dtype_*`, `api_inplace`, `api_scalar_input`, `api_large_shape`, `api_zero_dim`, `api_broadcast`

#### Fix 2: Relaxed block selection limits
**File:** `src/attest_cli/workflow/ascend_stages.py`

- `_default_analysis_plan()`: `block_limit: 3 → 6`
- `_select_target_blocks()`: removed `break` after first deferred block → now selects up to `block_limit` deferred blocks per epoch

#### Fix 3: Coverage build isolation
**File:** `src/attest_cli/workflow/ascend_stages.py` - `_run_for_root()`

- Added Phase 1.5: when compile fails, quarantine `*_attest.cpp` files and retry compile
- This allows extracting baseline coverage even when generated UT breaks the build
- Files restored after coverage extraction; logs document quarantine attempts

#### Fix 4: Resilient generate_code stage
**File:** `src/attest_cli/workflow/ascend_stages.py` - `execute()`

- LLM session failure no longer aborts entire stage → continues to next file
- Compile repair failure no longer aborts → logs warning and continues (isolation handled at coverage stage)
- Prevents single-file failure from killing entire epoch for all operators

#### Fix 5: Improved analysis feedback band width
**File:** `src/attest_cli/workflow/ascend_stages.py`

- `_coverage_failures()`: cap raised from `failures[:3]` → `failures[:6]`
- `_coverage_regression_failures()`: cap raised from `failures[:3]` → `failures[:6]`
- Allows analysis stage to surface more coverage gaps per epoch

---

## v0.3.x - Previous Releases
See git log for details.
