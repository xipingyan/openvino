// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class GatherDecompressFuseWeight : virtual public SubgraphBaseTest {
public:
    void run() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector params{
            std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape({-1, -1}))};
        auto convert1 = std::make_shared<ov::op::v0::Convert>(params[0], ov::element::i32);

        auto weights = op::v0::Constant::create(ov::element::u8, {4, 2}, {1, 2, 21, 22, 31, 32, 41, 42});
        auto convert2 = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);

        auto zp = op::v0::Constant::create(ov::element::u8, {4, 1}, {1, 1, 1, 1});
        auto convert3 = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);

        auto scale = op::v0::Constant::create(ov::element::f16, {4, 1}, {2, 2, 2, 2});

        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert2, convert3);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
        auto convert4 = std::make_shared<ov::op::v0::Convert>(multiply, ov::element::f32);

        auto axis  = op::v0::Constant::create(ov::element::i32, {1}, {0});

        auto gather = std::make_shared<ov::op::v8::Gather>(convert4, convert1, axis);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather)};
        function = std::make_shared<ov::Model>(results, params, "gather_decompress_fuse");

        std::vector<ov::Shape> input_shapes = {{{1, 1}, {2, 1}}};
        init_input_shapes(ov::test::static_shapes_to_test_representation(input_shapes));
        ov::test::SubgraphBaseTest::run();
    }
};

namespace {
TEST_F(GatherDecompressFuseWeight, smoke_GatherDecompressFuseWeight_CPU) {
    run();
}
}  // namespace
}  // namespace test
}  // namespace ov
