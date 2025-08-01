// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_copy_b_emitter.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/snippets/x64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::utils;

namespace ov::intel_cpu {

jit_brgemm_copy_b_emitter::jit_brgemm_copy_b_emitter(jit_generator_t* h,
                                                     cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr,
                                                     const snippets::KernelExecutorTablePtr& kernel_table,
                                                     const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto brgemm_repack = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_repack, "expects BrgemmCopyB node");

    // Note: even if the BrgemmCopyB node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmCopyBKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    const auto& brgemm_config = brgemm_repack->get_config();
    m_with_comp = brgemm_config.with_compensations();
    const BrgemmCopyBKernelConfig config(brgemm_config);
    m_kernel_executor = kernel_table->register_kernel<BrgemmCopyBKernelExecutor>(expr, compiled_kernel_cache, config);

    m_memory_offsets = {brgemm_repack->get_offset_in(), brgemm_repack->get_offset_out()};
    m_buffer_ids = {ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_output_port(0))};
    if (m_with_comp) {
        m_memory_offsets.push_back(brgemm_repack->get_offset_compensations());
        m_buffer_ids.push_back(ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_output_port(1)));
    }
}

void jit_brgemm_copy_b_emitter::validate_arguments(const std::vector<size_t>& in,
                                                   const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1, "expects 1 input");
    OV_CPU_JIT_EMITTER_ASSERT((m_with_comp && out.size() == 2) || (!m_with_comp && out.size() == 1),
                              "expects 2 outputs if there are compensations");
}

void jit_brgemm_copy_b_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], out[0]};
    if (out.size() > 1) {
        mem_ptrs_idxs.emplace_back(out[1]);
    }
    init_binary_call_regs(2, mem_ptrs_idxs);

    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    EmitABIRegSpills spill(h);
    spill.preamble(get_regs_to_spill());

    auto reserved_stack_size = sizeof(BrgemmCopyBKernel::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

    const std::vector<size_t> args_offsets{GET_OFF_BRGEMM_COPY_B_ARGS(src),
                                           GET_OFF_BRGEMM_COPY_B_ARGS(tr_src),
                                           GET_OFF_BRGEMM_COPY_B_ARGS(compensation_ptr)};
    const auto& mem_ptrs = ov::intel_cpu::utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        if (ov::snippets::utils::is_dynamic_value(m_memory_offsets[i])) {
            utils::push_ptr_with_runtime_offset_on_stack(h,
                                                         args_offsets[i],
                                                         mem_ptrs[i],
                                                         aux_reg,
                                                         GET_OFF(buffer_offsets) + m_buffer_ids[i] * sizeof(size_t));
        } else {
            utils::push_ptr_with_static_offset_on_stack(h, args_offsets[i], mem_ptrs[i], m_memory_offsets[i]);
        }
    }

    // No scratchpad => need to write nullptr manually
    if (!m_with_comp) {
        h->mov(h->qword[h->rsp + args_offsets.back()], reinterpret_cast<uintptr_t>(nullptr));
    }

    h->mov(aux_reg, reinterpret_cast<uintptr_t>(BrgemmCopyBKernelExecutor::execute));
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mov(abi_param2, h->rsp);

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    h->add(h->rsp, reserved_stack_size);

    spill.postamble();
}

}  // namespace ov::intel_cpu
