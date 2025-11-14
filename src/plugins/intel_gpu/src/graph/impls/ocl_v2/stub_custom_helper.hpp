// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "ocl_v2/primitive_ocl_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

inline cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                       cldnn::primitive_inst& instance,
                                       Stage& stage,
                                       const cldnn::kernel_arguments_desc& desc,
                                       const cldnn::kernel_arguments_data& args,
                                       bool needs_completion_event = false) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("StubCustomOptImpl::execute_stage"));
    cldnn::stream& stream = instance.get_network().get_stream();

    stream.set_arguments(*stage.kernel, desc, args);

    return stream.enqueue_kernel(*stage.kernel, desc, {}, events, needs_completion_event);
}

}  // namespace ov::intel_gpu::ocl