/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bitwise_not_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class BitwiseNotInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "BitwiseNotInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "BitwiseNotInferShape TearDown" << std::endl; }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
  size_t n = shape.GetDimNum();
  std::vector<int64_t> v(n, 0);
  for (size_t i = 0; i < n; i++) v[i] = shape.GetDim(i);
  return v;
}

// Test case for BOOL data type - basic functionality
TEST_F(BitwiseNotInferShape, case_01_bool_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{4, 4}, {4, 4}}, ge::DT_BOOL, ge::FORMAT_ND},   // Input: 4x4 boolean tensor
    },
    {
      {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for INT8 data type
TEST_F(BitwiseNotInferShape, case_02_int8_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{3, 5}, {3, 5}}, ge::DT_INT8, ge::FORMAT_ND},   // Input: 3x5 int8 tensor
    },
    {
      {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for INT16 data type
TEST_F(BitwiseNotInferShape, case_03_int16_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT16, ge::FORMAT_ND},   // Input: 2x3x4 int16 tensor
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for INT32 data type
TEST_F(BitwiseNotInferShape, case_04_int32_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{1, 6}, {1, 6}}, ge::DT_INT32, ge::FORMAT_ND},   // Input: 1x6 int32 tensor
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1, 6}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for INT64 data type
TEST_F(BitwiseNotInferShape, case_05_int64_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{7}, {7}}, ge::DT_INT64, ge::FORMAT_ND},   // Input: 7-element int64 tensor
    },
    {
      {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{7}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for UINT8 data type
TEST_F(BitwiseNotInferShape, case_06_uint8_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_UINT8, ge::FORMAT_ND},   // Input: 2x2x2x2 uint8 tensor
    },
    {
      {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 2, 2}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for higher dimensional tensor
TEST_F(BitwiseNotInferShape, case_07_high_dim) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_INT32, ge::FORMAT_ND},   // Input: 5D tensor
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5, 6}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for single element tensor
TEST_F(BitwiseNotInferShape, case_08_single_element) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{1}, {1}}, ge::DT_BOOL, ge::FORMAT_ND},   // Input: single element boolean tensor
    },
    {
      {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for zero-dim tensor (scalar)
TEST_F(BitwiseNotInferShape, case_09_scalar) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // Input: scalar int32 tensor
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{}};  // Empty vector for scalar
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for large tensor dimensions
TEST_F(BitwiseNotInferShape, case_10_large_dims) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{1000, 500}, {1000, 500}}, ge::DT_INT16, ge::FORMAT_ND},   // Input: large 2D tensor
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1000, 500}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for UINT16 (if supported)
TEST_F(BitwiseNotInferShape, case_11_uint16_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{3, 3}, {3, 3}}, ge::DT_UINT16, ge::FORMAT_ND},   // Input: 3x3 uint16 tensor
    },
    {
      {{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for UINT32 (if supported)
TEST_F(BitwiseNotInferShape, case_12_uint32_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{4, 2, 3}, {4, 2, 3}}, ge::DT_UINT32, ge::FORMAT_ND},   // Input: 4x2x3 uint32 tensor
    },
    {
      {{{}, {}}, ge::DT_UINT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for UINT64 (if supported)
TEST_F(BitwiseNotInferShape, case_13_uint64_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{5}, {5}}, ge::DT_UINT64, ge::FORMAT_ND},   // Input: 5-element uint64 tensor
    },
    {
      {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Additional edge cases for different formats
TEST_F(BitwiseNotInferShape, case_14_format_nd) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{2, 4, 8}, {2, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},   // Input: ND format
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Edge case: empty tensor
TEST_F(BitwiseNotInferShape, case_15_empty_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{0, 5}, {0, 5}}, ge::DT_INT32, ge::FORMAT_ND},   // Input: empty tensor (first dim is 0)
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{0, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Additional test cases to improve coverage

// Test case for UINT16 (if supported)
TEST_F(BitwiseNotInferShape, case_16_uint16_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{3, 3}, {3, 3}}, ge::DT_UINT16, ge::FORMAT_ND},   // Input: 3x3 uint16 tensor
    },
    {
      {{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for UINT32 (if supported)
TEST_F(BitwiseNotInferShape, case_17_uint32_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{4, 2, 3}, {4, 2, 3}}, ge::DT_UINT32, ge::FORMAT_ND},   // Input: 4x2x3 uint32 tensor
    },
    {
      {{{}, {}}, ge::DT_UINT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for UINT64 (if supported)
TEST_F(BitwiseNotInferShape, case_18_uint64_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{5}, {5}}, ge::DT_UINT64, ge::FORMAT_ND},   // Input: 5-element uint64 tensor
    },
    {
      {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for 1D tensor
TEST_F(BitwiseNotInferShape, case_19_onedim_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},   // Input: 1D tensor with 100 elements
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{100}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for very large dimensions
TEST_F(BitwiseNotInferShape, case_20_large_dims_edge) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{10000, 1000}, {10000, 1000}}, ge::DT_INT16, ge::FORMAT_ND},   // Input: large 2D tensor
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{10000, 1000}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for multi-dimensional edge case
TEST_F(BitwiseNotInferShape, case_21_multidim_edge) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{2, 1, 3, 1, 4}, {2, 1, 3, 1, 4}}, ge::DT_INT8, ge::FORMAT_ND},   // Input: 5D tensor with singleton dims
    },
    {
      {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 3, 1, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for single dimension with max value
TEST_F(BitwiseNotInferShape, case_22_max_single_dim) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseNot",
    {
      {{{2147483647LL}, {2147483647LL}}, ge::DT_INT32, ge::FORMAT_ND},   // Input: single huge dimension
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output: shape filled by InferShape
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2147483647LL}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

