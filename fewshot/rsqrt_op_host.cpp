#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class RsqrtInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "RsqrtInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "RsqrtInferShape TearDown" << std::endl; }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
  size_t n = shape.GetDimNum();
  std::vector<int64_t> v(n, 0);
  for (size_t i = 0; i < n; i++) v[i] = shape.GetDim(i);
  return v;
}

// Test case for basic functionality with different data types
TEST_F(RsqrtInferShape, case_01_valid_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for BF16 data type
TEST_F(RsqrtInferShape, case_02_valid_bf16) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},          // Input 1
    },
    {
      {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                         // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for FLOAT16 data type
TEST_F(RsqrtInferShape, case_03_valid_float16) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{5, 6}, {5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // Input 1
    },
    {
      {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                      // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{5, 6}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for scalar input
TEST_F(RsqrtInferShape, case_04_valid_scalar) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // Scalar input
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for large tensor
TEST_F(RsqrtInferShape, case_05_valid_large_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{100, 200, 300}, {100, 200, 300}}, ge::DT_FLOAT, ge::FORMAT_ND},  // Large tensor
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{100, 200, 300}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for single dimension tensor
TEST_F(RsqrtInferShape, case_06_valid_single_dim) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},               // Single dim tensor
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for 5D tensor
TEST_F(RsqrtInferShape, case_07_valid_5d_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},  // 5D tensor
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5, 6}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for empty tensor
TEST_F(RsqrtInferShape, case_08_valid_empty_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},               // Empty tensor (first dim is 0)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{0, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for 1D tensor with single element
TEST_F(RsqrtInferShape, case_09_valid_one_element) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // Single element
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case for 2D matrix
TEST_F(RsqrtInferShape, case_10_valid_matrix) {
  gert::InfershapeContextPara infershapeContextPara(
    "Rsqrt",
    {
      {{{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},           // Matrix
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{10, 20}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
