#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_bitwise_xor_scalar.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"

using namespace std;

class bitwise_xor_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "bitwise_xor_test SetUp" << endl; }
  static void TearDownTestCase() { cout << "bitwise_xor_test TearDown" << endl; }
};

TEST_F(bitwise_xor_test, case_default_float32) {
  auto self_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t scalar_value = 1;
  auto scalar_desc = ScalarDesc(scalar_value);
  auto out_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnBitwiseXorScalar, INPUT(self_desc, scalar_desc), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
