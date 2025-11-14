// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "primitive_type_base.h"
#include "program_node.h"
#include "stub_custom_inst.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(stub_custom_primitve)

// TODO: automatic register
// https://stackoverflow.com/questions/9459980/c-global-variable-not-initialized-when-linked-through-static-libraries-but-ok
struct custom_kernel_reference {
    std::unordered_map<std::string, custom_kernel_info> kernel_map;
    custom_kernel_reference() { CALL_REG_CUSTOM_KERNEL(WSAttention) };
} g_custom_kernel_reference;

void register_custom_kernel(const std::string& type, const custom_kernel_info& info) {
    g_custom_kernel_reference.kernel_map[type] = info;
}

typed_program_node<stub_custom_primitive>::typed_program_node(std::shared_ptr<stub_custom_primitive> prim, program& prog) : parent(prim, prog) {
    OPENVINO_ASSERT(g_custom_kernel_reference.kernel_map.count(prim->m_params["type"]), "custom kernel '", prim->m_params["type"], "' not support");
    const auto& info = g_custom_kernel_reference.kernel_map.at(prim->m_params["type"]);
    m_kernel = info.make_custom_kernel(prim, *this);
}

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout stub_inst::calc_output_layout(const stub_node& node, const kernel_impl_params& impl_param) {
    return node.m_kernel->calc_output_layout(node, impl_param);
}

template <typename ShapeType>
std::vector<layout> stub_inst::calc_output_layouts(const stub_node& node, const kernel_impl_params& impl_param) {
    return node.m_kernel->calc_output_layouts(node, impl_param);
}

template std::vector<layout> stub_inst::calc_output_layouts<ov::PartialShape>(const stub_node& node, const kernel_impl_params& impl_param);

std::string stub_inst::to_string(const stub_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite stubop_info;

    node_info->add("stub_custom_primitive info", stubop_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
stub_custom_primitive primitive is reusing memory with the input.
*/
stub_inst::typed_primitive_inst(network& network, const stub_node& node) : parent(network, node) {}

}  // namespace cldnn