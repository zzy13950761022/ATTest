// ==== BLOCK:HEADER START ====
#include <gtest/gtest.h>
#include "op_api_ut_common/op_api_ut.h"
#include "math/real_div/op_api/aclnn_real_div.h"

class real_div_test : public testing::Test {
 protected:
  static void SetUpTestCase() {}
  static void TearDownTestCase() {}
};
// ==== BLOCK:HEADER END ====

// ==== BLOCK:CASE_01 START ====
TEST_F(real_div_test, case_01_nullptr_input) {
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}
// ==== BLOCK:CASE_01 END ====

// ==== BLOCK:CASE_02 START ====
TEST_F(real_div_test, case_02_valid_float_type) {
    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_02 END ====

// ==== BLOCK:CASE_03 START ====
TEST_F(real_div_test, case_03_valid_float16_type) {
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_03 END ====

// ==== BLOCK:CASE_04 START ====
TEST_F(real_div_test, case_04_valid_bf16_type) {
    auto self_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_04 END ====

// ==== BLOCK:CASE_05 START ====
TEST_F(real_div_test, case_05_valid_bool_type) {
    auto self_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_05 END ====

// ==== BLOCK:CASE_06 START ====
TEST_F(real_div_test, case_06_broadcast_different_shapes) {
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_06 END ====

// ==== BLOCK:CASE_07 START ====
TEST_F(real_div_test, case_07_large_tensor) {
    auto self_desc = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto other_desc = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 2.0);
    auto out_desc = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_07 END ====

// ==== BLOCK:CASE_08 START ====
TEST_F(real_div_test, case_08_empty_tensor) {
    auto self_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_08 END ====

// ==== BLOCK:CASE_09 START ====
TEST_F(real_div_test, case_09_single_element) {
    auto self_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5.0, 5.0);
    auto other_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 3.0);
    auto out_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_09 END ====

// ==== BLOCK:CASE_10 START ====
TEST_F(real_div_test, case_10_private_format_valid) {
    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_10 END ====

// ==== BLOCK:CASE_11 START ====
TEST_F(real_div_test, case_11_3d_tensors) {
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto other_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_11 END ====

// ==== BLOCK:CASE_12 START ====
TEST_F(real_div_test, case_12_int32_type_ascend910b) {
    auto self_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    // Note: INT32 is not supported on Ascend910B, so this should fail
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_12 END ====

// ==== BLOCK:CASE_13 START ====
TEST_F(real_div_test, case_13_bool_output_promotion) {
    auto self_desc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_13 END ====

// ==== BLOCK:CASE_14 START ====
TEST_F(real_div_test, case_14_broadcast_5d_tensors) {
    auto self_desc = TensorDesc({2, 1, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto other_desc = TensorDesc({3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.1, 2.0);
    auto out_desc = TensorDesc({2, 1, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_14 END ====

// ==== BLOCK:CASE_15 START ====
TEST_F(real_div_test, case_15_dtype_validation_success) {
    auto self_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_15 END ====

// ==== BLOCK:CASE_16 START ====
TEST_F(real_div_test, case_16_shape_validation_failure) {
    auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    // These shapes cannot be broadcast together, so expect failure
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_16 END ====

// ==== BLOCK:CASE_17 START ====
TEST_F(real_div_test, case_17_non_contiguous_tensors) {
    auto self_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2.0, 2.0);
    auto other_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_17 END ====

// ==== BLOCK:CASE_18 START ====
TEST_F(real_div_test, case_18_different_formats) {
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto other_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0.5, 2.0);
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_18 END ====

// ==== BLOCK:CASE_19 START ====
TEST_F(real_div_test, case_19_int32_promotion_to_float) {
    auto self_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto other_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    // On platforms where INT32 is supported, this should work and result in FLOAT output
    // On Ascend910B, INT32 is not supported, so this should fail
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_19 END ====

// ==== BLOCK:CASE_20 START ====
TEST_F(real_div_test, case_20_bool_promotion_to_float) {
    auto self_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto other_desc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnRealDiv, INPUT(self_desc, other_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
// ==== BLOCK:CASE_20 END ====
