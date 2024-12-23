// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <memory>

#include "fully_connected_inst.h"
#include "impls/registry/implementation_manager.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/utils.hpp"

namespace cldnn {
namespace sycl_lz {

struct FullyConnectedImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("sycl_lz::fc")
    FullyConnectedImplementationManager(shape_types shape_type)
        : ImplementationManager(impl_types::sycl_lz, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());

        static const std::vector<format::type> supported_formats = {
            format::bfyx,
        };

        const auto& fc_node = node.as<fully_connected>();
        const auto& in_layout = fc_node.get_input_layout(0);
        const auto& out_layout = fc_node.get_output_layout(0);
        auto in0_dt = in_layout.data_type;
        auto wei_dt = fc_node.weights().get_output_layout(false).data_type;
        auto out_dt = out_layout.data_type;
        auto fc_prim = fc_node.get_primitive();

#if 1  // Sycl reference
        bool compressed_case = fc_prim->compressed_weights && one_of(in0_dt, {data_types::f16, data_types::f32}) &&
                               one_of(wei_dt, {data_types::u8, data_types::i8, data_types::u4, data_types::i4}) &&
                               one_of(out_dt, {data_types::f16, data_types::f32});
        if (compressed_case)
            return false;

        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats))
            return false;

        if (in_layout.data_padding || out_layout.data_padding)
            return false;

        GPU_DEBUG_LOG << "FullyConnectedImplementationManager::validate_impl" << std::endl;
        GPU_DEBUG_LOG << "in0_dt=" << in0_dt << ", wei_dt=" << wei_dt << ", out_dt=" << out_dt
                      << ", fc_prim->bias.empty()=" << fc_prim->bias.empty() << std::endl;
#else  // Sycl compressed_case
       // bool compressed_case = fc_prim->compressed_weights &&
       //                        one_of(in0_dt, {data_types::f16, data_types::f32}) &&
       //                        one_of(wei_dt, {data_types::u8, data_types::i8, data_types::u4, data_types::i4}) &&
       //                        one_of(out_dt, {data_types::f16, data_types::f32});
       // if (!compressed_case)
       //     return false;

        // if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value,
        // supported_formats))
        //     return false;

        // if (in_layout.data_padding || out_layout.data_padding)
        //     return false;
#endif
        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<fully_connected>());
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
