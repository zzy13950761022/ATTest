#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class BitwiseOrInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "BitwiseOrInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "BitwiseOrInferShape TearDown" << std::endl; }
};

// Test cases for bitwise_or infer shape
// ==== BLOCK:CASE_01 START ====
TEST_F(BitwiseOrInferShape, case_01_same_shape_int32) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
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
// ==== BLOCK:CASE_01 END ====

// ==== BLOCK:CASE_02 START ====
TEST_F(BitwiseOrInferShape, case_02_broadcast_different_shapes) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{1, 3}, {1, 3}}, ge::DT_INT32, ge::FORMAT_ND},   // Input 1
      {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_02 END ====

// ==== BLOCK:CASE_03 START ====
TEST_F(BitwiseOrInferShape, case_03_scalar_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},         // Input 1 (scalar)
      {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},            // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_03 END ====

// ==== BLOCK:CASE_04 START ====
TEST_F(BitwiseOrInferShape, case_04_different_dtypes_int8) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{4, 5}, {4, 5}}, ge::DT_INT8, ge::FORMAT_ND},    // Input 1
      {{{4, 5}, {4, 5}}, ge::DT_INT8, ge::FORMAT_ND},    // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},             // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_04 END ====

// ==== BLOCK:CASE_05 START ====
TEST_F(BitwiseOrInferShape, case_05_different_dtypes_uint16) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{3, 4, 5}, {3, 4, 5}}, ge::DT_UINT16, ge::FORMAT_ND}, // Input 1
      {{{3, 4, 5}, {3, 4, 5}}, ge::DT_UINT16, ge::FORMAT_ND}, // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_05 END ====

// ==== BLOCK:CASE_06 START ====
TEST_F(BitwiseOrInferShape, case_06_different_dtypes_int64) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND},   // Input 1
      {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND},   // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_06 END ====

// ==== BLOCK:CASE_07 START ====
TEST_F(BitwiseOrInferShape, case_07_different_dtypes_uint64) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_UINT64, ge::FORMAT_ND}, // Input 1
      {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_UINT64, ge::FORMAT_ND}, // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},          // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_07 END ====

// ==== BLOCK:CASE_08 START ====
TEST_F(BitwiseOrInferShape, case_08_complex_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{1, 1, 5}, {1, 1, 5}}, ge::DT_INT32, ge::FORMAT_ND}, // Input 1
      {{{3, 4, 1}, {3, 4, 1}}, ge::DT_INT32, ge::FORMAT_ND}, // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_08 END ====

// ==== BLOCK:CASE_09 START ====
TEST_F(BitwiseOrInferShape, case_09_different_dtypes_uint8) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{10}, {10}}, ge::DT_UINT8, ge::FORMAT_ND},       // Input 1
      {{{10}, {10}}, ge::DT_UINT8, ge::FORMAT_ND},       // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{10}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_09 END ====

// ==== BLOCK:CASE_10 START ====
TEST_F(BitwiseOrInferShape, case_10_different_dtypes_uint32) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{7, 8}, {7, 8}}, ge::DT_UINT32, ge::FORMAT_ND},  // Input 1
      {{{7, 8}, {7, 8}}, ge::DT_UINT32, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT32, ge::FORMAT_ND},          // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{7, 8}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_10 END ====

// ==== BLOCK:CASE_11 START ====
TEST_F(BitwiseOrInferShape, case_11_different_dtypes_int16) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_INT16, ge::FORMAT_ND}, // Input 1
      {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_INT16, ge::FORMAT_ND}, // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_11 END ====

// ==== BLOCK:CASE_12 START ====
TEST_F(BitwiseOrInferShape, case_12_large_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseOr",
    {
      {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND}, // Input 1 (scalar-like)
      {{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_INT32, ge::FORMAT_ND}, // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},           // Output (empty; filled by InferShape)
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5, 6}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_12 END ====
