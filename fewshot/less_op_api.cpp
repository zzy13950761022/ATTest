// ==== BLOCK:HEADER START ====
#include <gtest/gtest.h>
#include "math/less/op_api/aclnn_lt_tensor.h"
#include "math/less/op_api/aclnn_lt_scalar.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class test_aclnn_less : public testing::Test {
protected:
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}
};
// ==== BLOCK:HEADER END ====

// ==== BLOCK:CASE_01 START ====
TEST_F(test_aclnn_less, case_01_aclnnLtTensor_success) {
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_01 END ====

// ==== BLOCK:CASE_02 START ====
TEST_F(test_aclnn_less, case_02_aclnnLtTensor_different_dtypes) {
    auto self_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_02 END ====

// ==== BLOCK:CASE_03 START ====
TEST_F(test_aclnn_less, case_03_aclnnLtTensor_broadcast) {
    auto self_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_03 END ====

// ==== BLOCK:CASE_04 START ====
TEST_F(test_aclnn_less, case_04_aclnnLtScalar_success) {
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_scalar = ScalarDesc(1.0f);
    auto out_desc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtScalar, INPUT(self_desc, other_scalar), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_04 END ====

// ==== BLOCK:CASE_05 START ====
TEST_F(test_aclnn_less, case_05_aclnnLtScalar_int_types) {
    auto self_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_scalar = ScalarDesc(5);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtScalar, INPUT(self_desc, other_scalar), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_05 END ====

// ==== BLOCK:CASE_06 START ====
TEST_F(test_aclnn_less, case_06_aclnnInplaceLtTensor_success) {
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    
    auto ut = OP_API_UT(aclnnInplaceLtTensor, INPUT(self_desc, other_desc), OUTPUT());
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_06 END ====

// ==== BLOCK:CASE_07 START ====
TEST_F(test_aclnn_less, case_07_aclnnInplaceLtScalar_success) {
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_scalar = ScalarDesc(1.0f);
    
    auto ut = OP_API_UT(aclnnInplaceLtScalar, INPUT(self_desc, other_scalar), OUTPUT());
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_07 END ====

// ==== BLOCK:CASE_08 START ====
TEST_F(test_aclnn_less, case_08_aclnnLtTensor_bool_dtype) {
    auto self_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_08 END ====

// ==== BLOCK:CASE_09 START ====
TEST_F(test_aclnn_less, case_09_aclnnLtScalar_bool_dtype) {
    auto self_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto other_scalar = ScalarDesc(true);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtScalar, INPUT(self_desc, other_scalar), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_09 END ====

// ==== BLOCK:CASE_10 START ====
TEST_F(test_aclnn_less, case_10_aclnnLtTensor_uint_types) {
    auto self_desc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 100);
    auto other_desc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 100);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_10 END ====

// ==== BLOCK:CASE_11 START ====
TEST_F(test_aclnn_less, case_11_aclnnLtScalar_double_dtype) {
    auto self_desc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_scalar = ScalarDesc(1.5);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtScalar, INPUT(self_desc, other_scalar), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_11 END ====

// ==== BLOCK:CASE_12 START ====
TEST_F(test_aclnn_less, case_12_aclnnLtTensor_float16_dtype) {
    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    
    auto ut = OP_API_UT(aclnnLtTensor, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_12 END ====
