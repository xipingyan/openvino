// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_generator.hpp"

#include <xbyak/xbyak.h>

#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/plugin/x64/jit_conversion_emitters.hpp"
#include "emitters/plugin/x64/jit_dnnl_ext_emitters.hpp"
#include "emitters/plugin/x64/jit_eltwise_emitters.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/jit_brgemm_copy_b_emitter.hpp"
#include "emitters/snippets/x64/jit_brgemm_emitter.hpp"
#include "emitters/snippets/x64/jit_fill_emitter.hpp"
#include "emitters/snippets/x64/jit_horizon_emitter.hpp"
#include "emitters/snippets/x64/jit_kernel_emitter.hpp"
#include "emitters/snippets/x64/jit_loop_emitters.hpp"
#include "emitters/snippets/x64/jit_memory_emitters.hpp"
#include "emitters/snippets/x64/jit_reg_spill_emitters.hpp"
#include "emitters/snippets/x64/jit_snippets_emitters.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/xor.hpp"
#include "snippets/emitter.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/op/fill.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/perf_count.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/reg_spill.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/vector_buffer.hpp"
#include "snippets/runtime_configurator.hpp"
#include "snippets/target_machine.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"
#include "utils/general_utils.h"

#ifdef SNIPPETS_DEBUG_CAPS
#    include "emitters/snippets/x64/jit_debug_emitter.hpp"
#    include "emitters/snippets/x64/jit_perf_count_chrono_emitters.hpp"
#    include "emitters/snippets/x64/jit_perf_count_rdtsc_emitters.hpp"
#    include "emitters/snippets/x64/jit_segfault_detector_emitter.hpp"
#    include "emitters/snippets/x64/verbose.hpp"
#endif

#ifdef SNIPPETS_LIBXSMM_TPP
#    include "emitters/tpp/x64/jit_brgemm_emitter.hpp"
#    include "emitters/tpp/x64/jit_debug_emitter.hpp"
#    include "emitters/tpp/x64/jit_eltwise_emitters.hpp"
#    include "emitters/tpp/x64/jit_equation_emitter.hpp"
#    include "emitters/tpp/x64/jit_scalar_emitter.hpp"
#    include "transformations/tpp/common/op/brgemm.hpp"
#    include "transformations/tpp/common/op/modifiers.hpp"
#    include "transformations/tpp/x64/op/eltwise.hpp"
#    include "transformations/tpp/x64/op/equation.hpp"
#    include "transformations/tpp/x64/op/reduce.hpp"
#    include "transformations/tpp/x64/op/scalar.hpp"
// Note: for reference implementations
#    include <cmath>
#endif

namespace ov {

#ifdef SNIPPETS_DEBUG_CAPS
static bool is_load_emitter(const intel_cpu::jit_emitter* emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::jit_load_memory_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::jit_load_broadcast_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_store_emitter(const intel_cpu::jit_emitter* emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::jit_store_memory_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_segfault_detector_emitter(const intel_cpu::jit_emitter* emitter) {
    // default active for typical tensor memory access emitters
    bool ret = false;
    ret = is_load_emitter(emitter) || is_store_emitter(emitter) ||
          (dynamic_cast<const intel_cpu::jit_brgemm_emitter*>(emitter) != nullptr) ||
          (dynamic_cast<const intel_cpu::jit_brgemm_copy_b_emitter*>(emitter) != nullptr) ||
          (dynamic_cast<const intel_cpu::jit_kernel_emitter*>(emitter) != nullptr);
    return ret;
    // use below code to active all emitters for extend usage
    // return !dynamic_cast<const jit_nop_emitter*>(emitter);
}

#    define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                                 \
        {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> {            \
             auto emitter = std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__);                         \
             if (debug_config.enable_segfault_detector && is_segfault_detector_emitter(emitter.get())) {         \
                 auto segfault_emitter =                                                                         \
                     std::make_shared<jit_uni_segfault_detector_emitter>(h.get(),                                \
                                                                         isa,                                    \
                                                                         emitter.get(),                          \
                                                                         is_load_emitter(emitter.get()),         \
                                                                         is_store_emitter(emitter.get()),        \
                                                                         expr->get_node()->get_friendly_name()); \
                 return std::make_shared<jit_debug_emitter>(emitter,                                             \
                                                            segfault_emitter,                                    \
                                                            jit_debug_emitter::EmissionLocation::preamble);      \
             }                                                                                                   \
             return emitter;                                                                                     \
         },                                                                                                      \
         [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {                        \
             return e_type::get_supported_precisions(n);                                                         \
         }}
#else
#    define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                      \
        {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
             return std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__);                      \
         },                                                                                           \
         [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
             return e_type::get_supported_precisions(n);                                              \
         }}
#endif

#define CREATE_DEBUG_TPP_EMITTER(e_type)                                                               \
    {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> {      \
         return std::make_shared<DebugTppEmitter>(expr, std::make_shared<e_type>(h.get(), isa, expr)); \
     },                                                                                                \
     [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {                  \
         return e_type::get_supported_precisions(n);                                                   \
     }}
#define CREATE_CPU_EMITTER(e_type)                                                                                 \
    {[this]([[maybe_unused]] const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         return std::make_shared<e_type>(h.get(), isa, expr->get_node());                                          \
     },                                                                                                            \
     []([[maybe_unused]] const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
         return e_type::get_supported_precisions(n);                                                               \
     }}

#define CREATE_UNDEFINED_EMITTER(supported_precisions)                                                         \
    {[]([[maybe_unused]] const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         return nullptr;                                                                                       \
     },                                                                                                        \
     []([[maybe_unused]] const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {         \
         return supported_precisions;                                                                          \
     }}

class jit_snippet : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() override = default;

    jit_snippet() : jit_generator_t(jit_name()) {}

    void generate() override {}
};

intel_cpu::CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                              ov::intel_cpu::MultiCacheWeakPtr cache)
    : TargetMachine(std::make_shared<CPURuntimeConfigurator>(cache)),
      h(new jit_snippet()),
      isa(host_isa),
      compiled_kernel_cache(std::move(cache)) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::Buffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::VectorBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::RankNormalization::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::Reshape::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::Reorder::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);

    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[snippets::op::LoadReorder::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_broadcast_emitter);
    jitters[intel_cpu::LoadConvertSaturation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[intel_cpu::LoadConvertTruncation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);

    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_memory_emitter);
    jitters[intel_cpu::StoreConvertSaturation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_memory_emitter);
    jitters[intel_cpu::StoreConvertTruncation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_memory_emitter);

    jitters[snippets::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_scalar_emitter);
    jitters[snippets::op::BroadcastMove::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_broadcast_move_emitter);

    jitters[snippets::op::ConvertTruncation::get_type_info_static()] =
        CREATE_CPU_EMITTER(intel_cpu::jit_convert_truncation_emitter);
    jitters[snippets::op::ConvertSaturation::get_type_info_static()] =
        CREATE_CPU_EMITTER(intel_cpu::jit_convert_saturation_emitter);

    // ternary
    jitters[op::v1::Select::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_select_emitter);
    jitters[intel_cpu::FusedMulAdd::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_mul_add_emitter);

    // binary
    jitters[op::v1::Add::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_add_emitter);
    jitters[op::v1::Divide::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_divide_emitter);
    jitters[op::v1::Equal::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_equal_emitter);
    jitters[op::v1::FloorMod::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_floor_mod_emitter);
    jitters[op::v1::Greater::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_greater_emitter);
    jitters[op::v1::GreaterEqual::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_greater_equal_emitter);
    jitters[op::v1::Less::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_less_emitter);
    jitters[op::v1::LessEqual::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_less_equal_emitter);
    jitters[op::v1::LogicalAnd::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_and_emitter);
    jitters[op::v1::LogicalOr::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_or_emitter);
    jitters[op::v1::LogicalXor::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_xor_emitter);
    jitters[op::v1::Maximum::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_maximum_emitter);
    jitters[op::v1::Minimum::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_minimum_emitter);
    jitters[op::v1::Mod::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_mod_emitter);
    jitters[op::v1::Multiply::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_multiply_emitter);
    jitters[op::v1::NotEqual::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_not_equal_emitter);
    jitters[snippets::op::PowerStatic::get_type_info_static()] =
        CREATE_CPU_EMITTER(intel_cpu::jit_power_static_emitter);
    jitters[op::v1::Power::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_power_dynamic_emitter);
    jitters[op::v0::PRelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_prelu_emitter);
    jitters[op::v0::SquaredDifference::get_type_info_static()] =
        CREATE_CPU_EMITTER(intel_cpu::jit_squared_difference_emitter);
    jitters[op::v1::Subtract::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_subtract_emitter);
    jitters[op::v0::Xor::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_xor_emitter);

    // unary
    jitters[ov::op::v0::Abs::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_abs_emitter);
    jitters[ov::op::v0::Ceiling::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_ceiling_emitter);
    jitters[ov::op::v0::Clamp::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_clamp_emitter);
    jitters[ov::op::v0::Elu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_elu_emitter);
    jitters[ov::op::v0::Erf::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_erf_emitter);
    jitters[ov::op::v0::Exp::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_exp_emitter);
    jitters[ov::op::v0::Floor::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_floor_emitter);
    jitters[ov::op::v5::Round::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_round_emitter);
    jitters[ov::op::v1::LogicalNot::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_not_emitter);
    jitters[ov::op::v0::Negative::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_negative_emitter);
    jitters[ov::op::v0::Relu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_relu_emitter);
    jitters[ov::op::v0::Sigmoid::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_sigmoid_emitter);
    jitters[ov::op::v0::Sqrt::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_sqrt_emitter);
    jitters[ov::op::v0::Tanh::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_tanh_emitter);

    jitters[ov::intel_cpu::SwishNode::get_type_info_static()] = CREATE_CPU_EMITTER(ov::intel_cpu::jit_swish_emitter);
    jitters[ov::op::v4::HSwish::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_hswish_emitter);
    jitters[ov::op::v0::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_gelu_v0_emitter);
    jitters[ov::op::v7::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_gelu_v7_emitter);
    jitters[snippets::op::Fill::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_fill_emitter);

    jitters[snippets::op::HorizonMax::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_horizon_emitter);
    jitters[snippets::op::HorizonSum::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_horizon_emitter);

    // Note: jit_brgemm_emitter and jit_brgemm_copy_b_emitter support runtime recompilation, so their constructor takes
    // additional arguments
    jitters[intel_cpu::BrgemmCPU::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_brgemm_emitter,
                                configurator->get_kernel_executor_table(),
                                compiled_kernel_cache);
    jitters[intel_cpu::BrgemmCopyB::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_brgemm_copy_b_emitter,
                                configurator->get_kernel_executor_table(),
                                compiled_kernel_cache);
    jitters[snippets::op::ReduceMax::get_type_info_static()] = CREATE_UNDEFINED_EMITTER({{ov::element::f32}});
    jitters[snippets::op::ReduceSum::get_type_info_static()] = CREATE_UNDEFINED_EMITTER({{ov::element::f32}});
    // Service
    jitters[snippets::op::KernelStatic::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_kernel_static_emitter);
    jitters[snippets::op::KernelDynamic::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_kernel_dynamic_emitter);
    jitters[snippets::op::LoopBegin::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_loop_begin_emitter);
    jitters[snippets::op::LoopEnd::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_loop_end_emitter);
    jitters[snippets::op::RegSpillBegin::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_reg_spill_begin_emitter);
    jitters[snippets::op::RegSpillEnd::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_reg_spill_end_emitter);

#ifdef SNIPPETS_DEBUG_CAPS
    jitters[snippets::op::PerfCountBegin::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(ov::intel_cpu::jit_perf_count_chrono_start_emitter);
    jitters[snippets::op::PerfCountEnd::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(ov::intel_cpu::jit_perf_count_chrono_end_emitter);
    jitters[ov::intel_cpu::PerfCountRdtscBegin::get_type_info_static()] =
        CREATE_CPU_EMITTER(ov::intel_cpu::jit_perf_count_rdtsc_start_emitter);
    jitters[ov::intel_cpu::PerfCountRdtscEnd::get_type_info_static()] =
        CREATE_CPU_EMITTER(ov::intel_cpu::jit_perf_count_rdtsc_end_emitter);
#endif

#ifdef SNIPPETS_LIBXSMM_TPP
    jitters[intel_cpu::tpp::op::BrgemmTPP::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(BrgemmTppEmitter, configurator->get_kernel_executor_table(), compiled_kernel_cache);
    jitters[intel_cpu::tpp::op::Add::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Subtract::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Multiply::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Divide::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);

    jitters[intel_cpu::tpp::op::Exp::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    // Note: you can register Debug emitter for Unary/Binary operations as shown below:
    // jitters[intel_cpu::tpp::op::Add::get_type_info_static()] = CREATE_DEBUG_TPP_EMITTER(UnaryEltwiseTppEmitter);
    //
    // Note: you can register Reference emitter for Unary operations using std::function or lambda function as shown
    // below: jitters[intel_cpu::tpp::op::Exp::get_type_info_static()] =
    //        CREATE_SNIPPETS_EMITTER(ReferenceUnaryEltwiseTppEmitter, static_cast<float(*)(float)>(std::exp));
    // jitters[intel_cpu::tpp::op::Reciprocal::get_type_info_static()] =
    //         CREATE_SNIPPETS_EMITTER(ReferenceUnaryEltwiseTppEmitter, [](float x){ return 1.F/x; });
    jitters[intel_cpu::tpp::op::Reciprocal::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);

    jitters[intel_cpu::tpp::op::Relu::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Square::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::SquareRoot::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::ReduceMax::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(ReduceTppEmitter);
    jitters[intel_cpu::tpp::op::ReduceSum::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(ReduceTppEmitter);
    jitters[intel_cpu::tpp::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(ScalarTppEmitter);
    jitters[intel_cpu::tpp::op::EquationTPP::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(EquationTppEmitter);
#endif
}

std::shared_ptr<snippets::TargetMachine> intel_cpu::CPUTargetMachine::clone() const {
    const auto cloned = std::make_shared<intel_cpu::CPUTargetMachine>(isa, compiled_kernel_cache);
    cloned->configurator = std::make_shared<ov::snippets::RuntimeConfigurator>(*configurator);
    return cloned;
}

size_t intel_cpu::CPUTargetMachine::get_lanes() const {
    switch (isa) {
    case dnnl::impl::cpu::x64::avx2:
        return dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::avx2>::vlen / sizeof(float);
    case dnnl::impl::cpu::x64::sse41:
        return dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::sse41>::vlen / sizeof(float);
    case dnnl::impl::cpu::x64::avx512_core:
        return dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::avx512_core>::vlen / sizeof(float);
    default:
        OPENVINO_THROW("unknown isa ", isa);
    }
}

std::vector<snippets::Reg> intel_cpu::CPUTargetMachine::get_abi_arg_regs() const {
    const auto& abi_regs = dnnl::impl::cpu::x64::abi_param_regs;
    std::vector<snippets::Reg> res;
    for (const auto& r : abi_regs) {
        res.emplace_back(snippets::RegType::gpr, r);
    }
    return res;
}

std::vector<snippets::Reg> intel_cpu::CPUTargetMachine::get_gp_reg_pool() const {
    const auto num_gp_regs = 16;
    std::vector<snippets::Reg> reg_pool;
    for (size_t i = 0; i < num_gp_regs; i++) {
        if (none_of(i, Xbyak::Operand::RSP)) {
            reg_pool.emplace_back(snippets::RegType::gpr, i);
        }
    }
    return reg_pool;
}

std::vector<snippets::Reg> intel_cpu::CPUTargetMachine::get_vec_reg_pool() const {
    const auto num_vec_regs = [this]() {
        switch (isa) {
        case dnnl::impl::cpu::x64::avx2:
            return dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::avx2>::n_vregs;
        case dnnl::impl::cpu::x64::sse41:
            return dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::sse41>::n_vregs;
        case dnnl::impl::cpu::x64::avx512_core:
            return dnnl::impl::cpu::x64::cpu_isa_traits_t<dnnl::impl::cpu::x64::avx512_core>::n_vregs;
        default:
            OPENVINO_THROW("unknown isa ", isa);
        }
    }();
    std::vector<snippets::Reg> reg_pool;
    reg_pool.reserve(num_vec_regs);
    for (int i = 0; i < num_vec_regs; i++) {
        reg_pool.emplace_back(snippets::RegType::vec, static_cast<size_t>(i));
    }
    return reg_pool;
}

dnnl::impl::cpu::x64::cpu_isa_t intel_cpu::CPUTargetMachine::get_isa() const {
    return isa;
}

bool intel_cpu::CPUTargetMachine::is_supported() const {
    return dnnl::impl::cpu::x64::mayiuse(isa);
}

snippets::CompiledSnippetPtr intel_cpu::CPUTargetMachine::get_snippet() {
    OPENVINO_ASSERT(h->create_kernel() == dnnl::impl::status::success, "Failed to create jit_kernel in get_snippet()");
    const auto& result =
        std::make_shared<CompiledSnippetCPU>(std::unique_ptr<dnnl::impl::cpu::x64::jit_generator_t>(h.release()));
    // Note that we reset all the generated code, since it was copied into CompiledSnippetCPU
    h = std::make_unique<jit_snippet>();
    return result;
}

intel_cpu::CompiledSnippetCPU::CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::x64::jit_generator_t> h)
    : h_compiled(std::move(h)) {
    OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was nopt compiled");
}

const uint8_t* intel_cpu::CompiledSnippetCPU::get_code() const {
    return h_compiled->jit_ker();
}

size_t intel_cpu::CompiledSnippetCPU::get_code_size() const {
    return h_compiled->getSize();
}

bool intel_cpu::CompiledSnippetCPU::empty() const {
    return get_code_size() == 0;
}

intel_cpu::CPUGenerator::CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t host_isa, ov::intel_cpu::MultiCacheWeakPtr cache)
    : Generator(std::make_shared<CPUTargetMachine>(host_isa, std::move(cache))) {}
intel_cpu::CPUGenerator::CPUGenerator(const std::shared_ptr<CPUTargetMachine>& target) : Generator(target) {}

std::shared_ptr<snippets::Generator> intel_cpu::CPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target->clone());
    OPENVINO_ASSERT(cpu_target_machine,
                    "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<CPUGenerator>(cpu_target_machine);
}

ov::snippets::RegType intel_cpu::CPUGenerator::get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const {
    const auto op = out.get_node_shared_ptr();
    if (is_type_any_of<intel_cpu::BrgemmCPU, intel_cpu::BrgemmCopyB>(op)
#ifdef SNIPPETS_LIBXSMM_TPP
        || is_type<intel_cpu::tpp::op::Scalar>(op) ||
        std::dynamic_pointer_cast<intel_cpu::tpp::modifier::TensorProcessingPrimitive>(op)
#endif
    ) {
        return ov::snippets::RegType::gpr;
    }
    if (is_type_any_of<intel_cpu::FusedMulAdd, intel_cpu::SwishNode>(op)) {
        return ov::snippets::RegType::vec;
    }
    return ov::snippets::RegType::undefined;
}

bool intel_cpu::CPUGenerator::uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& e) const {
    bool need = std::dynamic_pointer_cast<intel_cpu::jit_brgemm_emitter>(e) ||
                std::dynamic_pointer_cast<intel_cpu::jit_brgemm_copy_b_emitter>(e);
#ifdef SNIPPETS_DEBUG_CAPS
    const auto cpu_target_machine = std::dynamic_pointer_cast<intel_cpu::CPUTargetMachine>(target);
    need = need || (cpu_target_machine && cpu_target_machine->debug_config.enable_segfault_detector) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_chrono_start_emitter>(e) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_chrono_end_emitter>(e) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_rdtsc_start_emitter>(e) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_rdtsc_end_emitter>(e);
#endif
#ifdef SNIPPETS_LIBXSMM_TPP
    need |= std::dynamic_pointer_cast<intel_cpu::ReferenceUnaryEltwiseTppEmitter>(e) ||
            std::dynamic_pointer_cast<intel_cpu::DebugTppEmitter>(e);
#endif
    return need;
}
}  // namespace ov
