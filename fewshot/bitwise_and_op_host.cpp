#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class BitwiseAndInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "BitwiseAndInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "BitwiseAndInferShape TearDown" << std::endl; }
};

// Test basic same shape scenario
TEST_F(BitwiseAndInferShape, case_01_same_shape_int32) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},   // Input 1
      {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test basic same shape scenario with different data types
TEST_F(BitwiseAndInferShape, case_02_same_shape_int16) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{4, 5}, {4, 5}}, ge::DT_INT16, ge::FORMAT_ND},   // Input 1
      {{{4, 5}, {4, 5}}, ge::DT_INT16, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test basic same shape scenario with uint16
TEST_F(BitwiseAndInferShape, case_03_same_shape_uint16) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{3, 4, 5}, {3, 4, 5}}, ge::DT_UINT16, ge::FORMAT_ND},   // Input 1
      {{{3, 4, 5}, {3, 4, 5}}, ge::DT_UINT16, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test basic same shape scenario with int64
TEST_F(BitwiseAndInferShape, case_04_same_shape_int64) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},   // Input 1
      {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test broadcasting scenario: scalar with tensor
TEST_F(BitwiseAndInferShape, case_05_scalar_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},          // Input 1 (scalar-like)
      {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test broadcasting scenario: 1D tensor with higher dimensional tensor
TEST_F(BitwiseAndInferShape, case_06_1d_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},          // Input 1 (1D)
      {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},   // Input 2 (3D)
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test broadcasting scenario: 2D tensor with 3D tensor
TEST_F(BitwiseAndInferShape, case_07_2d_3d_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 1 (2D)
      {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},   // Input 2 (3D)
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test broadcasting scenario: different compatible shapes
TEST_F(BitwiseAndInferShape, case_08_different_compatible_shapes) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{3, 1}, {3, 1}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 1
      {{{3, 4}, {3, 4}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test broadcasting scenario: complex broadcasting
TEST_F(BitwiseAndInferShape, case_09_complex_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{1, 1, 5}, {1, 1, 5}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 1
      {{{2, 3, 1}, {2, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test broadcasting scenario: 4D tensors
TEST_F(BitwiseAndInferShape, case_10_4d_tensors) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 1
      {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test broadcasting scenario: mixed dimensions
TEST_F(BitwiseAndInferShape, case_11_mixed_dimensions) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{2, 1, 4}, {2, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 1
      {{{1, 3, 1}, {1, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},    // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with unknown rank (-2) shapes
TEST_F(BitwiseAndInferShape, case_12_unknown_rank) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},        // Input 1 (unknown rank)
      {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},        // Input 2 (unknown rank)
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with dynamic shapes (-1 values)
TEST_F(BitwiseAndInferShape, case_13_dynamic_shapes) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{-1, 3}, {-1, 3}}, ge::DT_INT32, ge::FORMAT_ND},  // Input 1 (dynamic first dimension)
      {{{2, -1}, {2, -1}}, ge::DT_INT32, ge::FORMAT_ND},  // Input 2 (dynamic second dimension)
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with larger tensors
TEST_F(BitwiseAndInferShape, case_14_large_tensors) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{100, 200}, {100, 200}}, ge::DT_INT64, ge::FORMAT_ND},  // Input 1
      {{{100, 200}, {100, 200}}, ge::DT_INT64, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{100, 200}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with single element tensors
TEST_F(BitwiseAndInferShape, case_15_single_element) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},          // Input 1 (single element)
      {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},          // Input 2 (single element)
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with 5D tensors to ensure higher dimensional support
TEST_F(BitwiseAndInferShape, case_16_5d_tensors) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseAnd",
    {
      {{{1, 2, 1, 4, 1}, {1, 2, 1, 4, 1}}, ge::DT_UINT16, ge::FORMAT_ND},  // Input 1
      {{{3, 1, 5, 1, 6}, {3, 1, 5, 1, 6}}, ge::DT_UINT16, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 2, 5, 4, 6}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
