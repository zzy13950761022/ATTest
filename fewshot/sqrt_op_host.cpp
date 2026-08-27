/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SqrtInferShape : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "SqrtInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "SqrtInferShape TearDown" << std::endl;
    }
};

// Test case with FLOAT data type
TEST_F(SqrtInferShape, sqrt_infershape_test_0_float) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{2, 2, 1}, {2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with FLOAT16 data type
TEST_F(SqrtInferShape, sqrt_infershape_test_1_float16) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{3, 4, 5, 6}, {3, 4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with BF16 data type
TEST_F(SqrtInferShape, sqrt_infershape_test_2_bf16) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{-1, 2}, {-1, 2}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with single dimension
TEST_F(SqrtInferShape, sqrt_infershape_test_3_single_dim) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with scalar input
TEST_F(SqrtInferShape, sqrt_infershape_test_4_scalar) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with large tensor
TEST_F(SqrtInferShape, sqrt_infershape_test_5_large_tensor) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{1000, 500}, {1000, 500}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1000, 500}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with 1D tensor
TEST_F(SqrtInferShape, sqrt_infershape_test_6_1d_tensor) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with 4D tensor
TEST_F(SqrtInferShape, sqrt_infershape_test_7_4d_tensor) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{1, 3, 224, 224}, {1, 3, 224, 224}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 3, 224, 224}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with 5D tensor
TEST_F(SqrtInferShape, sqrt_infershape_test_8_5d_tensor) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{2, 4, 8, 16, 32}, {2, 4, 8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8, 16, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test case with unknown rank
TEST_F(SqrtInferShape, sqrt_infershape_test_9_unknown_rank) {
    gert::InfershapeContextPara infershapeContextPara("Sqrt",
                                                      {
                                                        {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
