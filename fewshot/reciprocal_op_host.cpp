#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ReciprocalInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "ReciprocalInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "ReciprocalInferShape TearDown" << std::endl; }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape) {
  size_t n = shape.GetDimNum();
  std::vector<int64_t> v(n, 0);
  for (size_t i = 0; i < n; i++) v[i] = shape.GetDim(i);
  return v;
}

// Test basic functionality with different data types and shapes
TEST_F(ReciprocalInferShape, case_01_valid_basic_float) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReciprocalInferShape, case_02_valid_basic_float16) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},      // Input 1
    },
    {
      {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReciprocalInferShape, case_03_valid_basic_bf16) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{5, 10}, {5, 10}}, ge::DT_BF16, ge::FORMAT_ND},             // Input 1
    },
    {
      {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                       // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{5, 10}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scalar input
TEST_F(ReciprocalInferShape, case_04_scalar_input) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                    // Scalar input
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test 1D tensor
TEST_F(ReciprocalInferShape, case_05_1d_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // 1D input
    },
    {
      {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{100}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test 2D tensor
TEST_F(ReciprocalInferShape, case_06_2d_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{32, 64}, {32, 64}}, ge::DT_BF16, ge::FORMAT_ND},           // 2D input
    },
    {
      {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                       // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{32, 64}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test 3D tensor
TEST_F(ReciprocalInferShape, case_07_3d_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},    // 3D input
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{8, 16, 32}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test 4D tensor
TEST_F(ReciprocalInferShape, case_08_4d_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // 4D input
    },
    {
      {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test large tensor
TEST_F(ReciprocalInferShape, case_09_large_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{1000, 1000}, {1000, 1000}}, ge::DT_FLOAT, ge::FORMAT_ND},  // Large 2D input
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1000, 1000}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test unknown rank scenario
TEST_F(ReciprocalInferShape, case_10_unknown_rank) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},                  // Unknown rank input
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};     // Should result in unknown rank
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test various formats (though only ND is supported according to def)
TEST_F(ReciprocalInferShape, case_11_format_nd) {
  gert::InfershapeContextPara infershapeContextPara(
    "Reciprocal",
    {
      {{{7, 14, 28}, {7, 14, 28}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // ND format
    },
    {
      {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{7, 14, 28}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test different combinations of data types
TEST_F(ReciprocalInferShape, case_12_mixed_dtypes) {
  // Test DT_FLOAT
  gert::InfershapeContextPara infershapeContextPara1(
    "Reciprocal",
    {
      {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    });
  std::vector<std::vector<int64_t>> expectOutputShape1 = {{64}};
  ExecuteTestCase(infershapeContextPara1, ge::GRAPH_SUCCESS, expectOutputShape1);

  // Test DT_FLOAT16
  gert::InfershapeContextPara infershapeContextPara2(
    "Reciprocal",
    {
      {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
      {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    });
  std::vector<std::vector<int64_t>> expectOutputShape2 = {{64}};
  ExecuteTestCase(infershapeContextPara2, ge::GRAPH_SUCCESS, expectOutputShape2);

  // Test DT_BF16
  gert::InfershapeContextPara infershapeContextPara3(
    "Reciprocal",
    {
      {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
      {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
    });
  std::vector<std::vector<int64_t>> expectOutputShape3 = {{64}};
  ExecuteTestCase(infershapeContextPara3, ge::GRAPH_SUCCESS, expectOutputShape3);
}
