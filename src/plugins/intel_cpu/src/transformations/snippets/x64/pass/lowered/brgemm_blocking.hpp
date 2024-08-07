// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface BrgemmBlocking
 * @brief Covers BrgemmCPU with blocking loops
 * @ingroup snippets
 */

class BrgemmBlocking : public snippets::lowered::pass::RangedPass {
public:
    OPENVINO_RTTI("BrgemmBlocking", "Pass")
    BrgemmBlocking();
    bool run(snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

    static snippets::lowered::SpecificIterationHandlers get_default_blocking_loop_handlers(size_t work_amount, size_t block_size);

private:
    static snippets::lowered::LinearIR::constExprIt move_new_memory_buffer(snippets::lowered::LinearIR& linear_ir,
                                                                           const snippets::lowered::LinearIR::constExprIt& brgemm_it);

    static snippets::lowered::LinearIR::constExprIt get_loop_begin_pos(snippets::lowered::LinearIR& linear_ir,
                                                                       const snippets::lowered::LinearIR::constExprIt& brgemm_it,
                                                                       bool include_repacking = true);
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov