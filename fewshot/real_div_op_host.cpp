/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_real_div_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

// ----------------RealDivInfershape-------------------
class RealDivInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RealDivInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RealDivInfershape TearDown" << std::endl;
    }
};

TEST_F(RealDivInfershape, real_div_infershape_test_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{2, 2, 1}, {2, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 3}, {2, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RealDivInfershape, real_div_infershape_test_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{3, 4, 5, 6, -1}, {3, 4, 5, 6, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 4, 5, 6, 1}, {3, 4, 5, 6, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 5, 6, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with different data types
TEST_F(RealDivInfershape, real_div_infershape_test_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RealDivInfershape, real_div_infershape_test_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 4, 1}, {1, 4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RealDivInfershape, real_div_infershape_test_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{2}, {2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RealDivInfershape, real_div_infershape_test_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with different combinations of data types between inputs
TEST_F(RealDivInfershape, real_div_infershape_test_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{3, 3}, {3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 3}, {1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // Expected output type based on operator def
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scalar division
TEST_F(RealDivInfershape, real_div_infershape_test_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // Scalar
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test with mismatched shapes that should still broadcast
TEST_F(RealDivInfershape, real_div_infershape_test_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{1, 5, 1}, {1, 5, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 1, 4}, {3, 1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Additional test cases to improve branch coverage

TEST_F(RealDivInfershape, real_div_infershape_test_9_empty_tensors)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{0, 2, 3}, {0, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 2, 3}, {0, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 2, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RealDivInfershape, real_div_infershape_test_10_single_element)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RealDivInfershape, real_div_infershape_test_11_large_dimensions)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{1000, 500}, {1000, 500}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 500}, {1, 500}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1000, 500},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RealDivInfershape, real_div_infershape_test_12_mixed_dtypes_edge_case)
{
    gert::InfershapeContextPara infershapeContextPara(
        "RealDiv",
        {
            {{{5, 5}, {5, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},  // Different dtype
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},  // Expected output type according to def
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
