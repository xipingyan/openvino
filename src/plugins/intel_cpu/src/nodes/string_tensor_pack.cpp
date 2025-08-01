// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_pack.h"

#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/string_tensor_pack.hpp"
#include "openvino/reference/string_tensor_pack.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {
StringTensorPack::StringTensorPack(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool StringTensorPack::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                            std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v15::StringTensorPack>(op)) {
            errorMessage = "Only opset15 StringTensorPack operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void StringTensorPack::getSupportedDescriptors() {
    // Validation is already done in the ov::opset15::StringTensorPack
}

void StringTensorPack::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    ov::element::Type indicesPrecision = getOriginalInputPrecisionAtPort(0);
    addSupportedPrimDesc({{LayoutType::ncsp, indicesPrecision},
                          {LayoutType::ncsp, indicesPrecision},
                          {LayoutType::ncsp, ov::element::u8}},
                         {{LayoutType::ncsp, ov::element::string}},
                         impl_desc_type::ref);
}

bool StringTensorPack::created() const {
    return getType() == Type::StringTensorPack;
}

bool StringTensorPack::needPrepareParams() const {
    return false;
}

void StringTensorPack::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

template <class T_idx>
void StringTensorPack::executeImpl() {
    const auto& data_shape = getSrcMemoryAtPort(0)->getStaticDims();
    ov::reference::string_tensor_pack(getSrcDataAtPortAs<const T_idx>(0),
                                      getSrcDataAtPortAs<const T_idx>(1),
                                      getSrcDataAtPortAs<const uint8_t>(2),
                                      getDstDataAtPortAs<std::string>(0),
                                      ov::shape_size(data_shape));
}

namespace {
struct StringTensorPackContext {
    StringTensorPack& node;
};
}  // namespace

template <typename T_idx>
struct StringTensorPack::StringTensorPackExecute {
    void operator()(StringTensorPackContext& ctx) {
        ctx.node.executeImpl<T_idx>();
    }
};

bool StringTensorPack::isExecutable() const {
    const bool port0_empty = isInputTensorAtPortEmpty(0);
    const bool port1_empty = isInputTensorAtPortEmpty(1);
    const bool any_empty = port0_empty || port1_empty;
    return !any_empty;
}

void StringTensorPack::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto indicesPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    StringTensorPackContext ctx = {*this};
    OV_SWITCH(intel_cpu,
              StringTensorPackExecute,
              ctx,
              indicesPrecision,
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::i64, int64_t))
}
}  // namespace ov::intel_cpu::node
