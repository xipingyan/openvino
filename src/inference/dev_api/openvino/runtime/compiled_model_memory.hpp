// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Helper API for compiled model memory management.
 * @file openvino/runtime/compiled_model_memory.hpp
 */

#pragma once

#include "openvino/runtime/compiled_model.hpp"

namespace ov {

/**
 * @brief Releases device-specific intermediate model weights memory for a compiled model.
 *
 * This is a thin dev API helper that forwards to CompiledModel::release_weights().
 * Currently it only works for the Intel GPU plugin, this triggers release of GPU USM weight buffers.
 */
inline void release_weights(ov::CompiledModel& compiled_model) {
    compiled_model.release_weights();
}

}  // namespace ov
