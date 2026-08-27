// ==== BLOCK:HEADER START ====
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class BitwiseXorInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "BitwiseXorInferShape SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "BitwiseXorInferShape TearDown" << std::endl; }
};
// ==== BLOCK:HEADER END ====

// ==== BLOCK:CASE_01 START ====
TEST_F(BitwiseXorInferShape, case_01_same_shape_int32) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
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
TEST_F(BitwiseXorInferShape, case_02_broadcast_simple) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},         // Input 1 - scalar-like
      {{{2, 3}, {2, 3}}, ge::DT_INT64, ge::FORMAT_ND},   // Input 2 - 2D tensor
    },
    {
      {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},            // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_02 END ====

// ==== BLOCK:CASE_03 START ====
TEST_F(BitwiseXorInferShape, case_03_broadcast_complex) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{1, 3, 1}, {1, 3, 1}}, ge::DT_INT16, ge::FORMAT_ND},  // Input 1
      {{{2, 1, 4}, {2, 1, 4}}, ge::DT_INT16, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},                 // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_03 END ====

// ==== BLOCK:CASE_04 START ====
TEST_F(BitwiseXorInferShape, case_04_different_dtypes_int8) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{4, 5}, {4, 5}}, ge::DT_INT8, ge::FORMAT_ND},    // Input 1
      {{{4, 5}, {4, 5}}, ge::DT_INT8, ge::FORMAT_ND},    // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},             // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_04 END ====

// ==== BLOCK:CASE_05 START ====
TEST_F(BitwiseXorInferShape, case_05_different_dtypes_uint16) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{1, 1}, {1, 1}}, ge::DT_UINT16, ge::FORMAT_ND},  // Input 1
      {{{3, 4}, {3, 4}}, ge::DT_UINT16, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND},           // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_05 END ====

// ==== BLOCK:CASE_06 START ====
TEST_F(BitwiseXorInferShape, case_06_different_dtypes_uint32) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{2, 2, 2}, {2, 2, 2}}, ge::DT_UINT32, ge::FORMAT_ND},  // Input 1
      {{{2, 2, 2}, {2, 2, 2}}, ge::DT_UINT32, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT32, ge::FORMAT_ND},                // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 2}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_06 END ====

// ==== BLOCK:CASE_07 START ====
TEST_F(BitwiseXorInferShape, case_07_different_dtypes_uint64) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{5}, {5}}, ge::DT_UINT64, ge::FORMAT_ND},       // Input 1
      {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},       // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},          // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_07 END ====

// ==== BLOCK:CASE_08 START ====
TEST_F(BitwiseXorInferShape, case_08_1d_broadcast) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},        // Input 1 - scalar
      {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},      // Input 2 - vector
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},          // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{10}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_08 END ====

// ==== BLOCK:CASE_09 START ====
TEST_F(BitwiseXorInferShape, case_09_4d_tensors) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},  // Input 1
      {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                       // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_09 END ====

// ==== BLOCK:CASE_10 START ====
TEST_F(BitwiseXorInferShape, case_10_broadcast_4d) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{1, 1, 1, 5}, {1, 1, 1, 5}}, ge::DT_UINT8, ge::FORMAT_ND},  // Input 1
      {{{2, 3, 4, 1}, {2, 3, 4, 1}}, ge::DT_UINT8, ge::FORMAT_ND},  // Input 2
    },
    {
      {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},                      // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_10 END ====

// ==== BLOCK:CASE_11 START ====
TEST_F(BitwiseXorInferShape, case_11_scalar_tensor) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},        // Input 1 - scalar
      {{{3, 3}, {3, 3}}, ge::DT_INT16, ge::FORMAT_ND},  // Input 2 - matrix
    },
    {
      {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},          // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{3, 3}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_11 END ====

// ==== BLOCK:CASE_12 START ====
TEST_F(BitwiseXorInferShape, case_12_empty_like_tensors) {
  gert::InfershapeContextPara infershapeContextPara(
    "BitwiseXor",
    {
      {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},        // Input 1 - empty tensor
      {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},        // Input 2 - empty tensor
    },
    {
      {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},          // Output
    });
  std::vector<std::vector<int64_t>> expectOutputShape = {{0}};
  ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// ==== BLOCK:CASE_12 END ====
