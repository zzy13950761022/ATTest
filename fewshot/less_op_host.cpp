/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_less_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class LessInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LessInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LessInfershape TearDown" << std::endl;
    }
};

TEST_F(LessInfershape, less_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
// Additional test cases for comprehensive coverage

TEST_F(LessInfershape, less_infer_shape_float)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_int32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_broadcast_same_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{1, 4}, {1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 1}, {3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_broadcast_different_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_scalar_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{5, 6}, {5, 6}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{5, 6}, {5, 6}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_int64)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_uint8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{7}, {7}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{7}, {7}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {7},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_double)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{2, 2, 2}, {2, 2, 2}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{2, 2, 2}, {2, 2, 2}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_bool)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{3, 5}, {3, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{3, 5}, {3, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_int8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LessInfershape, less_infer_shape_uint64)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Less",
        {
            {{{10}, {10}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{10}, {10}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
