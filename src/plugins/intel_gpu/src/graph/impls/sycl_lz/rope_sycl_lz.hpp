// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_inst.h"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/runtime/utils.hpp"

namespace cldnn {
namespace sycl_lz {

struct RoPEImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("sycl_lz::rope")
    RoPEImplementationManager(shape_types shape_type)
        : ImplementationManager(impl_types::sycl_lz, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<rope>());

        static const std::vector<format::type> supported_formats = {
            format::bfyx,
        };

        auto supported_types = {data_types::f32, data_types::f16};

        const auto& rope_node = node.as<rope>();
        const auto& in_layout = rope_node.get_input_layout(0);
        const auto& out_layout = rope_node.get_output_layout(0);

        if (rope_node.get_inputs_count() != 3u) {
            std::cout << "RoPEImplementationManager::validate_impl return false, rope_node.get_inputs_count() = " << rope_node.get_inputs_count() << std::endl;
            return false;
        }

        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats)) {
            std::cout << "RoPEImplementationManager::validate_impl return false, in_layout.format.value = " << in_layout.format.value << std::endl;
            return false;
        }

        if (!one_of(in_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            std::cout << "RoPEImplementationManager::validate_impl return false, in_layout.data_type=" << in_layout.data_type << std::endl;
            return false;
        }

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<rope>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;
        }
        out_fmts[0] = format::get_default_format(out_rank);

        return {in_fmts, out_fmts};
    }
};

}  // namespace sycl_lz
}  // namespace cldnn