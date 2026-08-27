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

using namespace ge;

class ExpTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ExpTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ExpTest TearDown" << std::endl;
    }
};

TEST_F(ExpTest, exp_infershape_diff_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_diff_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_same_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_same_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_bf16_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{2, 4, 8}, {2, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_single_element_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_large_shape_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{100, 200}, {100, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {100, 200},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_scalar_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Additional test cases for optional attributes
TEST_F(ExpTest, exp_infershape_with_base_attr_test)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"base", Ops::Math::AnyValue::CreateFrom<float>(2.71828f)}
    };
    
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_with_scale_attr_test)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"scale", Ops::Math::AnyValue::CreateFrom<float>(2.0f)}
    };
    
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_with_shift_attr_test)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"shift", Ops::Math::AnyValue::CreateFrom<float>(1.0f)}
    };
    
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{5, 6}, {5, 6}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_with_multiple_attrs_test)
{
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"base", Ops::Math::AnyValue::CreateFrom<float>(10.0f)},
        {"scale", Ops::Math::AnyValue::CreateFrom<float>(0.5f)},
        {"shift", Ops::Math::AnyValue::CreateFrom<float>(-1.0f)}
    };
    
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        attrs);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ExpTest, exp_infershape_different_dtypes_combinations)
{
    // Test with different data types
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    
    for (auto dtype : dtypes) {
        gert::InfershapeContextPara infershapeContextPara(
            "Exp",
            {
                {{{3, 3}, {3, 3}}, dtype, ge::FORMAT_ND},
            },
            {
                {{{}, {}}, dtype, ge::FORMAT_ND},
            });
        std::vector<std::vector<int64_t>> expectOutputShape = {
            {3, 3},
        };
        ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    }
}

TEST_F(ExpTest, exp_infershape_edge_cases)
{
    // Test with 1D tensor
    gert::InfershapeContextPara infershapeContextPara1D(
        "Exp",
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape1D = {
        {1024},
    };
    ExecuteTestCase(infershapeContextPara1D, ge::GRAPH_SUCCESS, expectOutputShape1D);
    
    // Test with 5D tensor
    gert::InfershapeContextPara infershapeContextPara5D(
        "Exp",
        {
            {{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape5D = {
        {2, 3, 4, 5, 6},
    };
    ExecuteTestCase(infershapeContextPara5D, ge::GRAPH_SUCCESS, expectOutputShape5D);
}

TEST_F(ExpTest, exp_infershape_empty_tensor_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Exp",
        {
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
