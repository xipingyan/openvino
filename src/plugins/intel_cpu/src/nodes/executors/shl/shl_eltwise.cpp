// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shl_eltwise.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "csinn/csi_nn.h"
#include "csinn_data_structure.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/shl/shl.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shl_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

static bool isEltwiseAlgorithmSupported(Algorithm algorithm) {
    return any_of(algorithm,
                  Algorithm::EltwiseAdd,
                  Algorithm::EltwiseSubtract,
                  Algorithm::EltwiseMultiply,
                  Algorithm::EltwiseDivide,
                  Algorithm::EltwiseMaximum,
                  Algorithm::EltwiseMinimum,
                  Algorithm::EltwiseExp,
                  Algorithm::EltwiseClamp,
                  Algorithm::EltwiseRelu,
                  Algorithm::EltwisePrelu);
}

bool ShlEltwiseExecutor::supports(const EltwiseConfig& config) {
    const auto& eltwiseAttrs = config.attrs;
    if (!isEltwiseAlgorithmSupported(eltwiseAttrs.data.algo)) {
        DEBUG_LOG("Eltwise algorithm ", algToString(eltwiseAttrs.data.algo), " is not supported");
        return false;
    }

    std::vector<MemoryDescPtr> srcDescs(config.descs.size() - 1);
    std::vector<MemoryDescPtr> dstDescs{config.descs.at(ARG_DST)};

    for (const auto& [argId, desc] : config.descs) {
        if (argId == ARG_DST) {
            continue;
        }

        srcDescs[argId - ARG_SRC] = desc;
    }

    constexpr auto supported_prec = ov::element::f32;
    auto is_precision_supported = [supported_prec](const MemoryDescPtr& desc) {
        return desc->getPrecision() == supported_prec;
    };
    if (!std::all_of(srcDescs.cbegin(), srcDescs.cend(), is_precision_supported) ||
        !std::all_of(dstDescs.cbegin(), dstDescs.cend(), is_precision_supported)) {
        DEBUG_LOG("ShlEltwise supports only f32");
        return false;
    }

    // check whether input and output layouts are equal
    if (srcDescs.front()->hasLayoutType(LayoutType::nCsp16c) || srcDescs.front()->hasLayoutType(LayoutType::nCsp8c)) {
        DEBUG_LOG("ShlEltwise does not support 'nCsp16c' or 'nCsp8c' layouts");
        return false;
    }
    const auto unifiedLayout = srcDescs.front()->hasLayoutType(LayoutType::ncsp) ? LayoutType::ncsp : LayoutType::nspc;
    const auto unifiedRank = srcDescs.front()->as<BlockedMemoryDesc>()->getBlockDims().size();
    auto has_unified_layout = [unifiedLayout, unifiedRank](const MemoryDescPtr& desc) {
        if (desc->hasLayoutType(LayoutType::nspc)) {  // ensure the same rank
            if (desc->as<BlockedMemoryDesc>()->getBlockDims().size() != unifiedRank) {
                return false;
            }
        }
        return desc->hasLayoutType(unifiedLayout);
    };
    if (!std::all_of(srcDescs.cbegin(), srcDescs.cend(), has_unified_layout) ||
        !std::all_of(dstDescs.cbegin(), dstDescs.cend(), has_unified_layout)) {
        DEBUG_LOG("ShlEltwise needs to ensure all inputs and outputs are in the same 'ncsp' or 'nspc' layouts");
        return false;
    }

    for (const auto& srcDesc : srcDescs) {
        csinn_layout_enum supportedLayout = getShlDataLayoutByMemoryDesc(srcDesc);
        switch (eltwiseAttrs.data.algo) {
        case Algorithm::EltwisePrelu:
            // SHL PRelu op only supports these two kinds of layout
            if (supportedLayout != csinn_layout_enum::CSINN_LAYOUT_NC1HWC0 &&
                supportedLayout != csinn_layout_enum::CSINN_LAYOUT_NCHW) {
                DEBUG_LOG("src descriptor layout is unsupported by SHL Prelu op: ", srcDesc->serializeFormat());
                return false;
            }
            break;
        default:
            if (supportedLayout == csinn_layout_enum::CSINN_LAYOUT_NULL) {
                DEBUG_LOG("src descriptor layout is unsupported by SHL: ", srcDesc->serializeFormat());
                return false;
            }
            continue;
        }
    }
    for (const auto& dstDesc : dstDescs) {
        if (getShlDataLayoutByMemoryDesc(dstDesc) == csinn_layout_enum::CSINN_LAYOUT_NULL) {
            DEBUG_LOG("dst descriptor layout is unsupported by SHL: ", dstDesc->serializeFormat());
            return false;
        }
    }

    return true;
}

ShlEltwiseExecutor::ShlEltwiseExecutor(EltwiseAttrs attrs,
                                       [[maybe_unused]] const MemoryArgs& memory,
                                       [[maybe_unused]] const ExecutorContext::CPtr& context)
    : shlEltwiseAttrs(std::move(attrs)) {}

bool ShlEltwiseExecutor::init(const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) {
    const auto& postOps = shlEltwiseAttrs.postOps;

    if (!postOps.empty()) {
        return false;
    }

    srcTensors = std::vector<ShlTensor>(srcDescs.size());
    dstTensors = std::vector<ShlTensor>(dstDescs.size());

    for (size_t i = 0; i < srcDescs.size(); i++) {
        srcTensors[i] = ShlTensor(sess,
                                  precisionToShlDataType(srcDescs[i]->getPrecision()),
                                  getShlDataLayoutByMemoryDesc(srcDescs[i]),
                                  srcDescs[i]->as<BlockedMemoryDesc>()->getBlockDims());
    }
    for (size_t i = 0; i < dstDescs.size(); i++) {
        dstTensors[i] = ShlTensor(sess,
                                  precisionToShlDataType(dstDescs[i]->getPrecision()),
                                  getShlDataLayoutByMemoryDesc(dstDescs[i]),
                                  dstDescs[i]->as<BlockedMemoryDesc>()->getBlockDims());
    }

    std::function<int()> initFunc = nullptr;
    enum csinn_api_enum shl_api = CSINN_RVV;
    switch (shlEltwiseAttrs.data.algo) {
    case Algorithm::EltwiseAdd:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_add_init(srcTensors[0].get(),
                                  srcTensors[1].get(),
                                  dstTensors[0].get(),
                                  static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_add(srcTensors[0].get(),
                             srcTensors[1].get(),
                             dstTensors[0].get(),
                             static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseSubtract:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_sub_init(srcTensors[0].get(),
                                  srcTensors[1].get(),
                                  dstTensors[0].get(),
                                  static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_sub(srcTensors[0].get(),
                             srcTensors[1].get(),
                             dstTensors[0].get(),
                             static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseMultiply:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_mul_init(srcTensors[0].get(),
                                  srcTensors[1].get(),
                                  dstTensors[0].get(),
                                  static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_mul(srcTensors[0].get(),
                             srcTensors[1].get(),
                             dstTensors[0].get(),
                             static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseDivide:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_div_init(srcTensors[0].get(),
                                  srcTensors[1].get(),
                                  dstTensors[0].get(),
                                  static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_div(srcTensors[0].get(),
                             srcTensors[1].get(),
                             dstTensors[0].get(),
                             static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseMaximum:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_maximum_init(srcTensors[0].get(),
                                      srcTensors[1].get(),
                                      dstTensors[0].get(),
                                      static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_maximum(srcTensors[0].get(),
                                 srcTensors[1].get(),
                                 dstTensors[0].get(),
                                 static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseMinimum:
        params = ov::intel_cpu::make_unique<ShlDisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_minimum_init(srcTensors[0].get(),
                                      srcTensors[1].get(),
                                      dstTensors[0].get(),
                                      static_cast<csinn_diso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_minimum(srcTensors[0].get(),
                                 srcTensors[1].get(),
                                 dstTensors[0].get(),
                                 static_cast<csinn_diso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseExp:
        params = ov::intel_cpu::make_unique<ShlSisoParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_exp_init(srcTensors[0].get(),
                                  dstTensors[0].get(),
                                  static_cast<csinn_siso_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_exp(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_siso_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseClamp:
        params = ov::intel_cpu::make_unique<ShlClipParams>(sess,
                                                           shl_api,
                                                           shlEltwiseAttrs.data.alpha,
                                                           shlEltwiseAttrs.data.beta);
        initFunc = [&]() {
            return csinn_clip_init(srcTensors[0].get(),
                                   dstTensors[0].get(),
                                   static_cast<csinn_clip_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_clip(srcTensors[0].get(), dstTensors[0].get(), static_cast<csinn_clip_params*>(params->get()));
        };
        break;
    case Algorithm::EltwiseRelu:
        if (shlEltwiseAttrs.data.alpha == 0) {
            params = ov::intel_cpu::make_unique<ShlReluParams>(sess, shl_api);
            initFunc = [&]() {
                return csinn_relu_init(srcTensors[0].get(),
                                       dstTensors[0].get(),
                                       static_cast<csinn_relu_params*>(params->get()));
            };
            shlExecFunc = [&]() {
                return csinn_relu(srcTensors[0].get(),
                                  dstTensors[0].get(),
                                  static_cast<csinn_relu_params*>(params->get()));
            };
        } else {
            params = ov::intel_cpu::make_unique<ShlReluParams>(sess, shl_api, shlEltwiseAttrs.data.alpha);
            initFunc = [&]() {
                return csinn_leaky_relu_init(srcTensors[0].get(),
                                             dstTensors[0].get(),
                                             static_cast<csinn_relu_params*>(params->get()));
            };
            shlExecFunc = [&]() {
                return csinn_leaky_relu(srcTensors[0].get(),
                                        dstTensors[0].get(),
                                        static_cast<csinn_relu_params*>(params->get()));
            };
        }
        break;
    case Algorithm::EltwisePrelu:
        params = ov::intel_cpu::make_unique<ShlPReluParams>(sess, shl_api);
        initFunc = [&]() {
            return csinn_prelu_init(srcTensors[0].get(),
                                    srcTensors[1].get(),
                                    dstTensors[0].get(),
                                    static_cast<csinn_prelu_params*>(params->get()));
        };
        shlExecFunc = [&]() {
            return csinn_prelu(srcTensors[0].get(),
                               srcTensors[1].get(),
                               dstTensors[0].get(),
                               static_cast<csinn_prelu_params*>(params->get()));
        };
        break;
    default:
        OPENVINO_THROW("Unsupported operation type for SHL Eltwise executor: ",
                       static_cast<int>(shlEltwiseAttrs.data.algo));
    }

    return initFunc != nullptr && initFunc() == CSINN_TRUE;
}

bool ShlEltwiseExecutor::update(const MemoryArgs& memory) {
    std::vector<MemoryDescPtr> srcDescs(memory.size() - 1);
    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            continue;
        }

        srcDescs[argId - ARG_SRC] = mem->getDescPtr();
    }

    std::vector<MemoryDescPtr> dstDescs{memory.at(ARG_DST)->getDescPtr()};

    return init(srcDescs, dstDescs);
}

void ShlEltwiseExecutor::execute(const MemoryArgs& memory) {
    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            continue;
        }

        const int i = argId - ARG_SRC;
        srcTensors[i].setData(mem->getData());
    }

    dstTensors[0].setData(memory.at(ARG_DST)->getData());

    assert(shlExecFunc);

    OPENVINO_ASSERT(shlExecFunc() == CSINN_TRUE, "ShlEltwiseExecutor: failed to execute");
}

}  // namespace ov::intel_cpu
