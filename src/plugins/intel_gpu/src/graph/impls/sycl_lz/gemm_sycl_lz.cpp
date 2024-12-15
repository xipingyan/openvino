// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_sycl_lz.hpp"
#include "gemm_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_sycl_lz_base.h"

#include <algorithm>
#include <memory>
namespace cldnn {
namespace sycl_lz {

struct gemm_sycl_lz : typed_primitive_sycl_lz_impl<gemm> {
    using parent = typed_primitive_sycl_lz_impl<gemm>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl_lz::gemm_sycl_lz)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_sycl_lz>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<gemm>& instance) override {
        GPU_DEBUG_LOG << "Not implemented[SYCL_RUNTIME]. GemmSyclLzImplementationManager::execute_impl" << std::endl;
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<gemm>();

        // auto& stream = downcast<ocl::sycl_stream>(network.get_stream());
        // auto& engine = downcast<ocl::sycl_engine>(network.get_engine());
        // ::sycl::context sycl_context = engine.get_sycl_context();
        // ::sycl::queue& sycl_queue = stream.get_sycl_queue();

        // const auto& params = instance.get_impl_params();
        // auto out_shape = params->output_layouts[0].get_shape();

        // auto output = instance.output_memory_ptr(0);
        // auto weights = instance.weights_memory();
        // auto bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        // std::vector<memory::ptr> inputs = { instance.input_memory_ptr(0) };
        // size_t in_id = instance.bias_term() ? 3 : 2;
        // if (!desc->decompression_scale.empty())
        //     inputs.push_back(instance.dep_memory_ptr(in_id++));

        // if (!desc->decompression_zero_point.empty())
        //     inputs.push_back(instance.dep_memory_ptr(in_id));

        // OPENVINO_ASSERT(!instance.bias_term() && !instance.get_node().has_fused_primitives());

        // ov::element::Type_t in_t = params->input_layouts[0].data_type;
        // ov::element::Type_t wei_t = params->weights_layout.value().data_type;
        // ov::element::Type_t out_t = params->output_layouts[0].data_type;
        // ov::element::Type_t ds_t = params->input_layouts[2].data_type;
        // ov::element::Type_t dzp_t = inputs.size() == 3 ? params->input_layouts[3].data_type : ov::element::Type_t::undefined;

        // OPENVINO_ASSERT(out_shape.size() == 3);
        // size_t M = out_shape[1];
        // size_t N = out_shape[2];
        // size_t K = params->weights_layout.value().get_partial_shape()[1].get_length();
        // size_t groups_num = params->input_layouts[2].get_shape()[1];
        // size_t group_size = K / groups_num;

        // OPENVINO_ASSERT(inputs.size() >= 2);

        // auto dzp_scalar = desc->decompression_zero_point_scalar;

        // bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

        // #define CASE(InputType, WeightsType, ZPType, ScaleType, DstType) \
        //     in_t == ov::element::InputType && \
        //     wei_t == ov::element::WeightsType && \
        //     out_t == ov::element::DstType && \
        //     ds_t == ov::element::ScaleType && \
        //     dzp_t == ov::element::ZPType

        // if ((CASE(f32, u4, f32, f32, f32)) || (CASE(f32, u4, undefined, f32, f32))) {
        //     const float* in = static_cast<const float*>(inputs[0]->buffer_ptr());
        //     const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
        //     float* out = static_cast<float*>(output->buffer_ptr());
        //     const float* ds = static_cast<const float*>(inputs[1]->buffer_ptr());
        //     const float* dzp = inputs.size() == 3 ? static_cast<const float*>(inputs[2]->buffer_ptr()) : nullptr;

        //     return to_ocl_event(stream, run_fc_int4_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        // } else if ((CASE(f16, u4, f16, f16, f16)) || (CASE(f16, u4, undefined, f16, f16))) {
        //     const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
        //     const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
        //     ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());
        //     const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
        //     const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;


        //     return to_ocl_event(stream, run_fc_int4_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        // } else if ((CASE(f16, u4, f16, f16, f32)) || (CASE(f16, u4, undefined, f16, f32))) {
        //     const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
        //     const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
        //     float* out = static_cast<float*>(output->buffer_ptr());
        //     const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
        //     const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;


        //     return to_ocl_event(stream, run_fc_int4_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        // } else if ((CASE(f32, u8, f32, f32, f32)) || (CASE(f32, u8, undefined, f32, f32))) {
        //     const float* in = static_cast<const float*>(inputs[0]->buffer_ptr());
        //     const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
        //     float* out = static_cast<float*>(output->buffer_ptr());
        //     const float* ds = static_cast<const float*>(inputs[1]->buffer_ptr());
        //     const float* dzp = inputs.size() == 3 ? static_cast<const float*>(inputs[2]->buffer_ptr()) : nullptr;

        //     return to_ocl_event(stream, run_fc_int8_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        // } else if ((CASE(f16, u8, f16, f16, f16)) || (CASE(f16, u8, undefined, f16, f16))) {
        //     const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
        //     const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
        //     ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());
        //     const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
        //     const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;

        //     return to_ocl_event(stream, run_fc_int8_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        // } else if ((CASE(f16, u8, f16, f16, f32)) || (CASE(f16, u8, undefined, f16, f32))) {
        //     const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
        //     const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
        //     float* out = static_cast<float*>(output->buffer_ptr());
        //     const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
        //     const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;

        //     return to_ocl_event(stream, run_fc_int8_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        // } else {
        //     OPENVINO_THROW("No instance for given types found: ", in_t, " ", wei_t, " ", out_t, " ", ds_t, " ", dzp_t);
        // }
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format::oiyx;
        return std::make_shared<WeightsReorderParams>(source_weights_layout, target_weights_layout);
    }

    static std::unique_ptr<primitive_impl> create(const gemm_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return cldnn::make_unique<gemm_sycl_lz>(engine, config, get_weights_reorder(impl_params));
    }
};

std::unique_ptr<primitive_impl> GemmSyclLzImplementationManager::create_impl(const program_node& node,
                                                                             const kernel_impl_params& params) const {
    GPU_DEBUG_LOG << "GemmSyclLzImplementationManager::create_impl" << std::endl;
    assert(node.is_type<gemm>());
    return sycl_lz::gemm_sycl_lz::create(static_cast<const gemm_node&>(node), params);
}

}  // namespace sycl_lz
}  // namespace cldnn
