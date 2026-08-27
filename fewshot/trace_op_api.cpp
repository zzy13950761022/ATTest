#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_trace.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class trace_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "trace_test SetUp" << endl; }
  static void TearDownTestCase() { cout << "trace_test TearDown" << endl; }
};

// Test case for nullptr input
TEST_F(trace_test, case_nullptr_input) {
  auto out_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnTrace, INPUT((aclTensor*)nullptr), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Test case for nullptr output
TEST_F(trace_test, case_nullptr_output) {
  auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT((aclTensor*)nullptr));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Test case for invalid dimension (not 2D)
TEST_F(trace_test, case_invalid_dim_not_2d) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);  // 3D tensor
  auto out_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);         // 0D tensor
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test case for invalid output dimension (not 0D)
TEST_F(trace_test, case_invalid_output_dim) {
  auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);    // 2D tensor
  auto out_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);        // 1D tensor (should be 0D)
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test case for valid 2x2 float32 matrix
TEST_F(trace_test, case_valid_2x2_float32) {
  auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for valid 3x3 float32 matrix
TEST_F(trace_test, case_valid_3x3_float32) {
  auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for float16
TEST_F(trace_test, case_valid_float16) {
  auto self_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for double
TEST_F(trace_test, case_valid_double) {
  auto self_desc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_desc = TensorDesc({}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.000001, 0.000001);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for int32
TEST_F(trace_test, case_valid_int32) {
  auto self_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto out_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);  // Output should be INT64 for integral inputs
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for int64
TEST_F(trace_test, case_valid_int64) {
  auto self_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto out_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);  // Output should be INT64 for integral inputs
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for int16
TEST_F(trace_test, case_valid_int16) {
  auto self_desc = TensorDesc({2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto out_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);  // Output should be INT64 for integral inputs
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for int8
TEST_F(trace_test, case_valid_int8) {
  auto self_desc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto out_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);  // Output should be INT64 for integral inputs
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for uint8
TEST_F(trace_test, case_valid_uint8) {
  auto self_desc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
  auto out_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);  // Output should be INT64 for integral inputs
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for bool
TEST_F(trace_test, case_valid_bool) {
  auto self_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND);  // Output should be INT64 for integral inputs
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for complex64
TEST_F(trace_test, case_valid_complex64) {
  auto self_desc = TensorDesc({2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for complex128
TEST_F(trace_test, case_valid_complex128) {
  auto self_desc = TensorDesc({2, 2}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto out_desc = TensorDesc({}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test case for BF16 (only supported on Ascend910B)
TEST_F(trace_test, case_valid_bf16) {
  auto self_desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
  auto out_desc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto ut = OP_API_UT(aclnnTrace, INPUT(self_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
