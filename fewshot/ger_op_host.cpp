#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class GerInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "GerInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "GerInferShape TearDown" << std::endl; }
};

// Test successful basic case with 1D inputs
TEST_F(GerInferShape, case_01_valid_basic) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (1D tensor)
      {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 2 (1D tensor)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{5, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with different data types - BF16
TEST_F(GerInferShape, case_02_valid_bf16) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{3}, {3}}, ge::DT_BF16, ge::FORMAT_ND},   // Input 1 (1D tensor)
      {{{7}, {7}}, ge::DT_BF16, ge::FORMAT_ND},   // Input 2 (1D tensor)
    },
    {
      {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},     // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 7}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with different data types - FLOAT16
TEST_F(GerInferShape, case_03_valid_float16) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // Input 1 (1D tensor)
      {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // Input 2 (1D tensor)
    },
    {
      {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{6, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with different shapes - larger dimensions
TEST_F(GerInferShape, case_04_valid_large_dims) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (1D tensor)
      {{{200}, {200}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 2 (1D tensor)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},         // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{100, 200}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with single element vectors
TEST_F(GerInferShape, case_05_valid_single_element) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (1D tensor with 1 element)
      {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 2 (1D tensor with 1 element)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test failure case: Input 1 is not 1D (2D input)
TEST_F(GerInferShape, case_06_invalid_input1_not_1d) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (2D tensor - invalid)
      {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},         // Input 2 (1D tensor - valid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test failure case: Input 2 is not 1D (2D input)
TEST_F(GerInferShape, case_07_invalid_input2_not_1d) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},         // Input 1 (1D tensor - valid)
      {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 2 (2D tensor - invalid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test failure case: Both inputs are not 1D
TEST_F(GerInferShape, case_08_invalid_both_inputs_not_1d) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (2D tensor - invalid)
      {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 2 (2D tensor - invalid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test with mixed data types
TEST_F(GerInferShape, case_09_valid_mixed_dtypes) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},       // Input 1 (FLOAT)
      {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},     // Input 2 (FLOAT16)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},         // Output (should match first input)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{5, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with 0-sized dimension (edge case)
TEST_F(GerInferShape, case_10_valid_zero_dim) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},       // Input 1 (0-sized 1D tensor)
      {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},       // Input 2 (normal 1D tensor)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},         // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{0, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with large single dimension
TEST_F(GerInferShape, case_11_valid_large_single_dim) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{1000}, {1000}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (large 1D tensor)
      {{{2000}, {2000}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 2 (large 1D tensor)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1000, 2000}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with 3D input (should fail - not 1D)
TEST_F(GerInferShape, case_12_invalid_3d_input) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (3D tensor - invalid)
      {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},               // Input 2 (1D tensor - valid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test with various combinations of dimension mismatches to cover all branches in the condition
TEST_F(GerInferShape, case_13_invalid_input1_0d_input2_1d) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Input 1 (0D tensor - invalid)
      {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},      // Input 2 (1D tensor - valid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test with input1 1D and input2 0D to cover another branch combination
TEST_F(GerInferShape, case_14_invalid_input1_1d_input2_0d) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},      // Input 1 (1D tensor - valid)
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Input 2 (0D tensor - invalid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test with both inputs as 0D tensors
TEST_F(GerInferShape, case_15_invalid_both_inputs_0d) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Input 1 (0D tensor - invalid)
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Input 2 (0D tensor - invalid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test with 4D input (should fail - not 1D)
TEST_F(GerInferShape, case_16_invalid_4d_input) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},   // Input 1 (4D tensor - invalid)
      {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},                      // Input 2 (1D tensor - valid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test with 1D inputs of different data types to ensure consistent behavior
TEST_F(GerInferShape, case_17_valid_different_dtypes_consistency) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{7}, {7}}, ge::DT_BF16, ge::FORMAT_ND},       // Input 1 (BF16)
      {{{9}, {9}}, ge::DT_FLOAT16, ge::FORMAT_ND},    // Input 2 (FLOAT16)
    },
    {
      {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},         // Output should match first input type
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{7, 9}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with very small dimensions (just to ensure edge case handling)
TEST_F(GerInferShape, case_18_valid_minimal_dimensions) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},      // Input 1 (1D with 1 element)
      {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},      // Input 2 (1D with 1 element)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with negative-like scenario (though negative dims shouldn't occur in practice)
// Using zero dimensions which is a valid edge case
TEST_F(GerInferShape, case_19_valid_zero_dimensions_variations) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},      // Input 1 (0-sized 1D tensor)
      {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},      // Input 2 (0-sized 1D tensor)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},        // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{0, 0}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Additional test to ensure all branches in the main condition are covered
TEST_F(GerInferShape, case_20_invalid_input1_2d_input2_3d) {
  gert::InfershapeContextPara infershapeContextPara(
    "Ger",
    {
      {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},     // Input 1 (2D tensor - invalid)
      {{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}, // Input 2 (3D tensor - invalid)
    },
    {
      {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},              // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
