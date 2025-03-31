// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_sycl_lz.hpp"

#include "primitive_sycl_lz_base.h"
#include "rope_inst.h"
#include "runtime/sycl_lz/sycl_lz_engine.hpp"
#include "runtime/sycl_lz/sycl_lz_stream.hpp"

namespace cldnn {
namespace sycl_lz {

#define DEBUG_POS(txt) std::cout << "== " << __FILE__ << ":" << __LINE__ << " " << txt << std::endl

template <typename Type>
::sycl::event run_rope_common(::sycl::queue& queue,
                              bool enqueue_barrier,
                              const Type* input,
                              const Type* cos,
                              const Type* sin,
                              Type* dst,
                              size_t group_size,
                              size_t groups_num,
                              const ov::Shape& out_shape) {
    if (enqueue_barrier) {
        queue.submit([=](::sycl::handler& cgh) {
            cgh.ext_oneapi_barrier();
        });
    }

    return queue.submit([=](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<3>(out_shape[0], out_shape[1], out_shape[2]), [=](::sycl::id<3> index) {
#define VEC_SIZE          16
#define HALF_ROTARY_NDIMS 64
            const uint32_t b = index[0];
            const uint32_t h = index[1];
            // const uint32_t p = index[2] * VEC_SIZE / HALF_ROTARY_NDIMS;
            // const uint32_t r = index[2]* VEC_SIZE) % HALF_ROTARY_NDIMS;

            // uint input_idx = INPUT0_GET_INDEX(b, h, p, 0);
            // uint input_idx = (
            //     (1*0) +
            //     (128*(shape_info[8])) +
            //     ((128*(28 + (shape_info[8] + shape_info[9])))*0) +
            //     ((128*(28 + (shape_info[8] + shape_info[9]))*1)*0) +
            //     ((128*(28 + (shape_info[8] + shape_info[9]))*1*1*1*1)*0) +
            //     ((128*(28 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0))*0)
            //     ) +
            //     (0)*1 +
            //     (h)*128 +
            //     (p)*(128*(28 + (shape_info[8] + shape_info[9]))*1*1*1*1) +
            //     (b)*(128*(28 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0));

            // uint input_idx = (
            //     (1*0) +
            //     (128*(shape_info[8])) +
            //     ((128*(4 + (shape_info[8] + shape_info[9])))*0) +
            //     ((128*(4 + (shape_info[8] + shape_info[9]))*1)*0) +
            //     ((128*(4 + (shape_info[8] + shape_info[9]))*1*1*1*1)*0) +
            //     ((128*(4 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0))*0)
            //     ) +
            //     (0)*1 +
            //     (h)*128 +
            //     (p)*(128*(4 + (shape_info[8] + shape_info[9]))*1*1*1*1) +
            //     (b)*(128*(4 + (shape_info[8] + shape_info[9]))*1*1*1*1*(shape_info[1] + 0));
            // out << "== dst id:" << dst_index << ", accumulator=" << accumulator << sycl::endl;
        });
    });
}

struct rope_impl : typed_primitive_sycl_lz_impl<rope> {
    using parent = typed_primitive_sycl_lz_impl<rope>;
    using parent::parent;
    // using kernel_selector_t = kernel_selector::rope_kernel_selector;
    // using kernel_params_t = kernel_selector::rope_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl_lz::rope_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<rope_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        // parent::load(ib);
        // if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
        //     auto& kernel_selector = kernel_selector_t::Instance();
        //     auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
        //     kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        // }
        DEBUG_POS("Not implemented.");
    }

    // static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
    //     const auto& primitive = impl_param.typed_desc<rope>();
    //     auto params = get_default_params<kernel_selector::rope_params>(impl_param, is_shape_agnostic);

    //     params.head_cnt = primitive->config.head_cnt;
    //     params.head_size = primitive->config.head_size;
    //     params.rotary_ndims = primitive->config.rotary_ndims;
    //     params.gather_rank = primitive->gather_rank;

    //     params.slice_start = primitive->config.slice_start;
    //     params.slice_stop = primitive->config.slice_stop;

    //     params.axis = primitive->config.is_qwen || primitive->config.is_chatglm ? 2 : 3;
    //     params.num_of_inputs = primitive->config.is_chatglm || (primitive->config.output_trans0213 && primitive->config.is_interleaved)  ? 2 : 3;

    //     if (params.gather_rank > 0) {
    //         params.num_of_inputs++;
    //     }

    //     params.is_qwen = primitive->config.is_qwen;
    //     params.is_chatglm = primitive->config.is_chatglm;
    //     params.is_interleaved = primitive->config.is_interleaved;
    //     params.support_2d_rope = primitive->config.support_2d_rope;
    //     params.transposed_input = primitive->config.input_trans0213;

    //     for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
    //         params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
    //     }
    //     return params;
    // }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        const auto& primitive = impl_params.typed_desc<rope>();

        if (primitive->config.is_chatglm || primitive->config.is_qwen) {
            return primitive_impl::static_canonicalize_shapes(impl_params);
        } else {
            auto updated_impl_params = canonicalize_fused_shapes(impl_params);

            std::set<size_t> canonicalize_from_begin = {1, 2};
            for (size_t i = 0; i < updated_impl_params.input_layouts.size(); ++i) {
                auto& input_layout = updated_impl_params.input_layouts[i];
                if (canonicalize_from_begin.count(i) != 0) {
                    input_layout.set_partial_shape(extend_shape_to_rank_from_begin(input_layout.get_partial_shape()));
                } else {
                    input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_layout.get_partial_shape()));
                }
            }

            auto& output_layout = updated_impl_params.output_layouts[0];
            output_layout.set_partial_shape(extend_shape_to_rank_from_end(output_layout.get_partial_shape()));

            return updated_impl_params;
        }
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    static std::unique_ptr<primitive_impl> create(const rope_node& arg, const kernel_impl_params& impl_params) {
        GPU_DEBUG_LOG << "rope_impl::create" << std::endl;
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return std::make_unique<rope_impl>(engine, config);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */, typed_primitive_inst<rope>& instance) override {
        GPU_DEBUG_LOG << "SYCL Kernel: rope_impl::execute_impl" << std::endl;
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<rope>();

        auto& stream = downcast<sycl_lz::sycl_lz_stream>(network.get_stream());
        auto& engine = downcast<sycl_lz::sycl_lz_engine>(network.get_engine());
        ::sycl::context sycl_context = engine.get_sycl_context();
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();

        const auto& params = instance.get_impl_params();
        auto out_shape = params->output_layouts[0].get_shape();

        auto output_ptr = instance.output_memory_ptr(0);
        auto input_ptr = instance.input_memory_ptr(0);
        auto cos_ptr = instance.input_memory_ptr(1);
        auto sin_ptr = instance.input_memory_ptr(2);

        ov::element::Type_t in_t = params->input_layouts[0].data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;

// #define CASE(IO_Type) in_t == ov::element::IO_Type&& out_t == ov::element::IO_Type
//         if (CASE(f32)) {
//             run_rope_common(sycl_queue,
//                             false,
//                             reinterpret_cast<f32>(input_ptr->buffer_ptr()),
//                             reinterpret_cast<f32>(cos_ptr->buffer_ptr()),
//                             reinterpret_cast<f32>(sin_ptr->buffer_ptr()),
//                             reinterpret_cast<f32>(output_ptr->buffer_ptr()));
//         } else if ((CASE(f16, f16, f32))) {
//         } else {
//         }
        return stream.create_base_event(sycl::event());
    }
};

std::unique_ptr<primitive_impl> RoPEImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return sycl_lz::rope_impl::create(static_cast<const rope_node&>(node), params);
}

}  // namespace sycl_lz
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sycl_lz::rope_impl)
// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rope)
