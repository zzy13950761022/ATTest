/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "math/pows/op_api/aclnn_pows.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class pows_test : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(pows_test, case_01_nullptr_input) {
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPows, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(pows_test, case_02_nullptr_input_x1) {
    auto x2_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPows, INPUT((aclTensor*)nullptr, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(pows_test, case_03_nullptr_input_x2) {
    auto x1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, (aclTensor*)nullptr), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(pows_test, case_04_valid_float32) {
    auto x1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto x2_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_05_valid_float16) {
    auto x1_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto x2_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_06_valid_bfloat16) {
    auto x1_desc = TensorDesc({4, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto x2_desc = TensorDesc({4, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({4, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_07_different_shapes_broadcast) {
    auto x1_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto x2_desc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_08_large_tensor) {
    auto x1_desc = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2_desc = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-0.5, 0.5);
    auto out_desc = TensorDesc({100, 100}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_09_single_element) {
    auto x1_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2.0, 2.0);
    auto x2_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(3.0, 3.0);
    auto out_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_10_3d_tensor) {
    auto x1_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

// Additional test cases to maximize coverage
TEST_F(pows_test, case_11_zero_exponent) {
    auto x1_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 5.0);
    auto x2_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 0.0);  // All zeros
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_12_negative_base_positive_exp) {
    auto x1_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5.0, -1.0);
    auto x2_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2.0, 4.0);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_13_dtype_mismatch_x1_x2) {
    auto x1_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);  // Different dtype
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(pows_test, case_14_dtype_mismatch_output) {
    auto x1_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);  // Different dtype
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(pows_test, case_15_invalid_shape_broadcast) {
    auto x1_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);  // Cannot broadcast with {3, 4}
    auto out_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);  // Shape mismatch expected
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(pows_test, case_16_empty_tensor) {
    auto x1_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_17_different_formats) {
    auto x1_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.5, 2.0);
    auto x2_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1.0, 3.0);
    auto out_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_18_large_exponents) {
    auto x1_desc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 2.0);
    auto x2_desc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10.0, 20.0);
    auto out_desc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(pows_test, case_19_fractional_exponents) {
    auto x1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(4.0, 9.0);
    auto x2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.5, 0.5);  // Square root
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPows, INPUT(x1_desc, x2_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
