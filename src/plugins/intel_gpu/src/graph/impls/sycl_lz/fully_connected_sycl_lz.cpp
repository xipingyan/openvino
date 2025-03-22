// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_sycl_lz.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "fully_connected_inst.h"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_sycl_lz_base.h"
#include "runtime/sycl_lz/sycl_lz_engine.hpp"
#include "runtime/sycl_lz/sycl_lz_stream.hpp"

namespace cldnn {
namespace sycl_lz {

template <typename A, typename B>
struct AccumulatorType {
    using type = float;
};

template<> struct AccumulatorType<::sycl::half, ::sycl::half> {
    using type = ::sycl::half;
};

template <typename AType, typename WType, typename DType>
::sycl::event run_fc_common(::sycl::queue& queue,
                         bool enqueue_barrier,
                         const AType* a,
                         const WType* w,
                         DType* dst,
                         size_t M,
                         size_t N,
                         size_t K,
                         size_t group_size,
                         size_t groups_num,
                         const ov::Shape& out_shape) {
    GPU_DEBUG_LOG << "Pure SYCL kernel: run_fc_common, AType=" << typeid(AType).name()
                  << ", WType=" << typeid(AType).name() << ", DType=" << typeid(DType).name() << ", M=" << M
                  << ", N=" << N << ", K=" << K << std::endl;
    if (enqueue_barrier) {
        queue.submit([=](::sycl::handler& cgh) {
            cgh.ext_oneapi_barrier();
        });
    }

    return queue.submit([=](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<3>(out_shape[0], out_shape[1], out_shape[2]), [=](::sycl::id<3> index) {
            const uint32_t b = index[0];
            const uint32_t m = index[1];
            const uint32_t n = index[2];

            using accum_t = typename AccumulatorType<AType, WType>::type;
            accum_t accumulator = 0.0f;

            for (uint32_t y = 0; y < K; ++y) {
                const uint32_t input0_offset = y + m * K + b * M * K;
                // const uint32_t zp_offset = (y / group_size % groups_num) * N + n % N;
                // const uint32_t decomp_offset = (y / group_size % groups_num) * N + n % N;
                const uint32_t filter_offset = y + n * K;

                // accum_t zp_val = has_value ? static_cast<accum_t>(dzp_value) : static_cast<accum_t>(zp[zp_offset]);
                // accum_t scale = s[decomp_offset];
                accum_t filter_compressed = static_cast<accum_t>(w[filter_offset]);
                // accum_t filter_val = (filter_compressed) * scale;
                accumulator += a[input0_offset] * filter_compressed;
                // out << "  == accumulator=" << accumulator << ", a[input0_offset]=" << a[input0_offset]
                //     << ", filter_compressed=" << filter_compressed << ", K=" << K << ", a[0]=" << a[0] << sycl::endl;
            }
            const uint32_t dst_index = n + m * N + b * N * M;
            // accumulator = 20;
            dst[dst_index] = accumulator;
            // out << "== dst id:" << dst_index << ", accumulator=" << accumulator << sycl::endl;
        });
    });
}

inline std::string usm_alloc_2_str(sycl::usm::alloc mem_type) {
    switch (mem_type) {
    case sycl::usm::alloc::device:
        return "sycl::usm::alloc::device";
    case sycl::usm::alloc::host:
        return "sycl::usm::alloc::host";
    case sycl::usm::alloc::shared:
        return "sycl::usm::alloc::shared";
    case sycl::usm::alloc::unknown:
        return "sycl::usm::alloc::unknown";
    }
    return std::string();
}

template <typename AType, typename WType, typename DType>
::sycl::event run_fc_in_f16_out_f32(::sycl::queue& queue,
                                    bool enqueue_barrier,
                                    const AType* a,
                                    const WType* w,
                                    DType* dst,
                                    size_t M,
                                    size_t N,
                                    size_t K,
                                    size_t group_size,
                                    size_t groups_num,
                                    const ov::Shape& out_shape) {
    GPU_DEBUG_LOG << "Temp solution. GemmSyclLzImplementationManager::run_fc_in_f16_out_f32" << std::endl;
    if (enqueue_barrier) {
        queue.submit([=](::sycl::handler& cgh) {
            cgh.ext_oneapi_barrier();
        });
    }

    GPU_DEBUG_LOG << " == a ptr type:" << usm_alloc_2_str(sycl::get_pointer_type(a, queue.get_context())) << std::endl;
    GPU_DEBUG_LOG << " == w ptr type:" << usm_alloc_2_str(sycl::get_pointer_type(w, queue.get_context())) << std::endl;

    return queue.submit([=](::sycl::handler& cgh) {
        // Print inside SYCL Kernel.
        // sycl::stream out(384 * 20, 1024, cgh);
        cgh.parallel_for(::sycl::range<3>(out_shape[0], out_shape[1], out_shape[2]), [=](::sycl::id<3> index) {
            const uint32_t b = index[0];
            const uint32_t m = index[1];
            const uint32_t n = index[2];
            using accum_t = typename AccumulatorType<AType, WType>::type;
            accum_t accumulator = 0.0f;

            for (uint32_t y = 0; y < K; ++y) {
                const uint32_t input0_offset = y + m * K + b * M * K;
                // const uint32_t zp_offset = (y / group_size % groups_num) * N + n % N;
                // const uint32_t decomp_offset = (y / group_size % groups_num) * N + n % N;
                const uint32_t filter_offset = y + n * K;

                // accum_t zp_val = has_value ? static_cast<accum_t>(dzp_value) : static_cast<accum_t>(zp[zp_offset]);
                // accum_t scale = s[decomp_offset];
                accum_t filter_compressed = static_cast<accum_t>(w[filter_offset]);
                // accum_t filter_val = (filter_compressed) * scale;
                accumulator += a[input0_offset] * filter_compressed;
                // out << "  == accumulator=" << accumulator << ", a[input0_offset]=" << a[input0_offset]
                //     << ", filter_compressed=" << filter_compressed << ", K=" << K << sycl::endl;
            }
            const uint32_t dst_index = n + m * N + b * N * M;
            dst[dst_index] = accumulator;
            // out << "== dst id:" << dst_index << ", accumulator=" << accumulator << sycl::endl;
        });
    });
}

struct fully_connected_sycl_lz : typed_primitive_sycl_lz_impl<fully_connected> {
    using parent = typed_primitive_sycl_lz_impl<fully_connected>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl_lz::fully_connected_sycl_lz)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<fully_connected_sycl_lz>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(fully_connected_inst& instance) const {
        GPU_DEBUG_LOG << "Not Implemented. get_arguments" << std::endl;
        return std::unordered_map<int, dnnl::memory>();
        // std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        // {
        //     auto weights = instance.weights_memory();
        //     auto offset = sycl_lz::get_offset(instance.get_input_layout(1),
        //     _pd.dnnl::primitive_desc_base::weights_desc(0)); args.insert({DNNL_ARG_WEIGHTS,
        //     weights->get_sycl_lz_memory(_pd.weights_desc(0), offset)});
        // }

        // if (instance.bias_term()) {
        //     auto bias = instance.bias_memory();
        //     auto offset = sycl_lz::get_offset(instance.get_input_layout(2),
        //     _pd.dnnl::primitive_desc_base::weights_desc(1)); args.insert({DNNL_ARG_BIAS,
        //     bias->get_sycl_lz_memory(_pd.weights_desc(1), offset)});
        // }

        // const auto& prim = instance.get_impl_params()->typed_desc<fully_connected>();
        // if (prim->compressed_weights) {
        //     const auto weights_dt = instance.get_input_layout(1).data_type;
        //     auto weight_bitwidth = ov::element::Type(weights_dt).bitwidth();
        //     OPENVINO_ASSERT(weight_bitwidth == 8 || weight_bitwidth == 4, "[GPU] oneDNN supports only 4bit/8bit
        //     compressed weights"); int idx = prim->bias.empty() ? 2 : 3;

        //     if (!prim->decompression_scale.empty()) {
        //         auto decompression_scale_idx = idx++;
        //         auto scale_mem = instance.dep_memory_ptr(decompression_scale_idx);
        //         dnnl::memory::desc desc = sycl_lz::layout_to_memory_desc(scale_mem->get_layout(),
        //         dnnl::memory::format_tag::a, true); args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
        //         scale_mem->get_sycl_lz_memory(desc)});
        //     }

        //     if (!prim->decompression_zero_point.empty()) {
        //         auto decompression_zp_idx = idx++;
        //         auto zp_mem = instance.dep_memory_ptr(decompression_zp_idx);
        //         dnnl::memory::desc desc = sycl_lz::layout_to_memory_desc(zp_mem->get_layout(),
        //         dnnl::memory::format_tag::a, true); args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
        //         zp_mem->get_sycl_lz_memory(desc)});
        //     }

        //     if (prim->activation_scale.is_valid()) {
        //         auto activation_scale_idx = idx++;
        //         auto act_scale_mem = instance.dep_memory_ptr(activation_scale_idx);
        //         // TODO: handle group_size here
        //         dnnl::memory::desc desc = sycl_lz::layout_to_memory_desc(act_scale_mem->get_layout(),
        //         dnnl::memory::format_tag::a, true); args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
        //         act_scale_mem->get_sycl_lz_memory(desc)});
        //     }
        // }

        // return args;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format::oiyx;

        return std::make_shared<WeightsReorderParams>(source_weights_layout, target_weights_layout);
    }

    // static void transform_layouts(layout& input_layout,
    //                               layout& weights_layout,
    //                               layout& output_layout,
    //                               size_t prim_input_size) {
    //     auto input_pshape = input_layout.get_partial_shape();
    //     auto weights_pshape = weights_layout.get_partial_shape();

    //     size_t input_size = (prim_input_size > input_pshape.size()) ? input_pshape.size() : prim_input_size;
    //     int64_t feature = input_pshape[std::min(input_size, static_cast<size_t>(4)) - 1].get_length();
    //     if (input_size == 3) {
    //         feature = std::max({input_layout.spatial(0), input_layout.spatial(1), input_layout.spatial(2)});
    //     }

    //     if (input_size > 3) {
    //         input_layout.set_partial_shape(reshape_to_2d(input_pshape, feature));
    //     }
    //     if (weights_pshape.size() != 2) {
    //         weights_layout.set_partial_shape(reshape_to_2d(weights_pshape, feature));
    //     }
    //     if (input_size == 3) {
    //         output_layout.set_partial_shape({input_layout.batch(), input_layout.feature(), weights_layout.batch(),
    //         1});
    //     } else {
    //         output_layout.set_partial_shape({input_layout.batch(), weights_layout.batch()});
    //     }

    //     if (input_size == 3) {
    //         combine_bf_with_first_spatial_dim(input_layout);
    //         combine_bf_with_first_spatial_dim(output_layout);
    //     }
    // }

    // static std::shared_ptr<dnnl::inner_product_forward::primitive_desc>
    //     get_inner_product_primitive_descriptor(const kernel_impl_params& impl_params,
    //                                            cldnn::engine& engine,
    //                                            size_t prim_input_size,
    //                                            bool has_bias,
    //                                            const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
    //     auto input_layout = impl_params.get_input_layout(0);
    //     auto weights_layout = impl_params.get_input_layout(1);
    //     auto output_layout = impl_params.get_output_layout();

    //     transform_layouts(input_layout, weights_layout, output_layout, prim_input_size);

    //     auto input_md = sycl_lz::layout_to_memory_desc(input_layout, dnnl::memory::format_tag::undef, false);
    //     auto weights_md = sycl_lz::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any);
    //     auto output_md = sycl_lz::layout_to_memory_desc(output_layout, dnnl::memory::format_tag::ab, false);

    //     if (has_bias) {
    //         auto bias_md = sycl_lz::layout_to_memory_desc(impl_params.get_input_layout(2),
    //         dnnl::memory::format_tag::any, true); return
    //         std::make_shared<dnnl::inner_product_forward::primitive_desc>(
    //             engine.get_sycl_lz_engine(),
    //             dnnl::prop_kind::forward_inference,
    //             input_md,
    //             weights_md,
    //             bias_md,
    //             output_md,
    //             attr);
    //     } else {
    //         return std::make_shared<dnnl::inner_product_forward::primitive_desc>(
    //             engine.get_sycl_lz_engine(),
    //             dnnl::prop_kind::forward_inference,
    //             input_md,
    //             weights_md,
    //             output_md,
    //             attr);
    //     }
    // }

    // static std::shared_ptr<dnnl::matmul::primitive_desc>
    //     get_matmul_primitive_descriptor(const kernel_impl_params& impl_params,
    //                                     cldnn::engine& engine,
    //                                     size_t prim_input_size,
    //                                     bool has_bias,
    //                                     const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
    //     auto input_layout = impl_params.get_input_layout(0);
    //     auto weights_layout = impl_params.get_input_layout(1);
    //     auto output_layout = impl_params.get_output_layout();

    //     transform_layouts(input_layout, weights_layout, output_layout, prim_input_size);

    //     auto input_md = sycl_lz::layout_to_memory_desc(input_layout, dnnl::memory::format_tag::ab, false);
    //     // TODO: should change format to any. May need a reorder.
    //     auto weights_md = sycl_lz::layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::ba);
    //     auto output_md = sycl_lz::layout_to_memory_desc(output_layout, dnnl::memory::format_tag::ab, false);

    //     if (has_bias) {
    //         auto bias_md = sycl_lz::layout_to_memory_desc(impl_params.get_input_layout(2),
    //         dnnl::memory::format_tag::ab, false); return std::make_shared<dnnl::matmul::primitive_desc>(
    //             engine.get_sycl_lz_engine(),
    //             input_md,
    //             weights_md,
    //             bias_md,
    //             output_md,
    //             attr);
    //     } else {
    //         return std::make_shared<dnnl::matmul::primitive_desc>(
    //             engine.get_sycl_lz_engine(),
    //             input_md,
    //             weights_md,
    //             output_md,
    //             attr);
    //     }
    // }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<fully_connected>& instance) override {
        GPU_DEBUG_LOG << "Temp solution. GemmSyclLzImplementationManager::execute_impl" << std::endl;
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<fully_connected>();

        auto& stream = downcast<sycl_lz::sycl_lz_stream>(network.get_stream());
        auto& engine = downcast<sycl_lz::sycl_lz_engine>(network.get_engine());
        ::sycl::context sycl_context = engine.get_sycl_context();
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();

        const auto& params = instance.get_impl_params();
        auto out_shape = params->output_layouts[0].get_shape();

        auto output = instance.output_memory_ptr(0);
        auto weights = instance.weights_memory();
        GPU_DEBUG_LOG << "Have bias: instance.bias_term()=" << instance.bias_term() << std::endl;
        auto bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        std::vector<memory::ptr> inputs = {instance.input_memory_ptr(0)};
        size_t in_id = instance.bias_term() ? 3 : 2;
        if (!desc->decompression_scale.empty())
            inputs.push_back(instance.dep_memory_ptr(in_id++));

        if (!desc->decompression_zero_point.empty())
            inputs.push_back(instance.dep_memory_ptr(in_id));

        OPENVINO_ASSERT(!instance.bias_term() && !instance.get_node().has_fused_primitives());

        ov::element::Type_t in_t = params->input_layouts[0].data_type;
        ov::element::Type_t wei_t = params->weights_layout.value().data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;
        // if (bias) {
        //     ov::element::Type_t ds_t = params->input_layouts[2].data_type;
        //     ov::element::Type_t dzp_t =
        //         inputs.size() == 3 ? params->input_layouts[3].data_type : ov::element::Type_t::undefined;
        // }

        OPENVINO_ASSERT(out_shape.size() == 3);
        size_t M = out_shape[1];
        size_t N = out_shape[2];
        size_t K = params->weights_layout.value().get_partial_shape()[1].get_length();
        size_t groups_num = bias ? params->input_layouts[2].get_shape()[1] : 1;
        size_t group_size = K / groups_num;

        OPENVINO_ASSERT(inputs.size() >= 1);

        bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

#define CASE(InputType, WeightsType, DstType) \
    in_t == ov::element::InputType&& wei_t == ov::element::WeightsType&& out_t == ov::element::DstType

        if (CASE(f32, f32, f32)) {
            const float* in = static_cast<const float*>(inputs[0]->buffer_ptr());
            const float* wei = static_cast<const float*>(weights->buffer_ptr());
            float* out = static_cast<float*>(output->buffer_ptr());
            return stream.create_base_event(
                run_fc_common(sycl_queue, barrier, in, wei, out, M, N, K, group_size, groups_num, out_shape));
        } else if ((CASE(f16, f16, f32))) {
            const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
            const ::sycl::half* wei = static_cast<const ::sycl::half*>(weights->buffer_ptr());
            float* out = static_cast<float*>(output->buffer_ptr());

            return stream.create_base_event(
                run_fc_common(sycl_queue, barrier, in, wei, out, M, N, K, group_size, groups_num, out_shape));
        } else if ((CASE(f16, f16, f16))) {
            const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
            const ::sycl::half* wei = static_cast<const ::sycl::half*>(weights->buffer_ptr());
            ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());

            return stream.create_base_event(
                run_fc_common(sycl_queue, barrier, in, wei, out, M, N, K, group_size, groups_num, out_shape));
        } else {
            OPENVINO_THROW("No instance for given types found: ", in_t, " ", wei_t, " ", out_t);
        }
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
        GPU_DEBUG_LOG << "Not implemented. fully_connected_sycl_lz::save" << std::endl;
    }

    void load(BinaryInputBuffer& ib) override {
        GPU_DEBUG_LOG << "Not implemented. fully_connected_sycl_lz::load" << std::endl;
    }

    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg,
                                                  const kernel_impl_params& impl_params) {
        GPU_DEBUG_LOG << "fully_connected_sycl_lz::create" << std::endl;
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return std::make_unique<fully_connected_sycl_lz>(engine, config, get_weights_reorder(impl_params));
    }
};

std::unique_ptr<primitive_impl> FullyConnectedImplementationManager::create_impl(
    const program_node& node,
    const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return sycl_lz::fully_connected_sycl_lz::create(static_cast<const fully_connected_node&>(node), params);
}

}  // namespace sycl_lz
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sycl_lz::fully_connected_sycl_lz)
