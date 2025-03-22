// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace cldnn {

/// @brief Defines available engine types
enum class engine_types : int32_t {
    ocl,
    sycl,    // sycl is based on OpenCL backend.
    sycl_lz  // sycl_lz is based on LevelZero backend.
};

inline std::ostream& operator<<(std::ostream& os, const engine_types& type) {
    switch (type) {
    case engine_types::ocl: os << "ocl"; break;
    case engine_types::sycl: os << "sycl"; break;
    case engine_types::sycl_lz: os << "sycl_lz"; break;
    default: os << "unknown"; break;
    }

    return os;
}

/// @brief Defines available runtime types
enum class runtime_types : int32_t {
    ocl,
    sycl_lz,
};

inline std::ostream& operator<<(std::ostream& os, const runtime_types& type) {
    switch (type) {
    case runtime_types::ocl: os << "ocl"; break;
    case runtime_types::sycl_lz: os << "sycl_lz"; break;
    default: os << "unknown"; break;
    }

    return os;
}

}  // namespace cldnn
