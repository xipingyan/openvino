// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/rope.hpp"

namespace cldnn {
namespace sycl_lz {
void register_implementations();

namespace detail {

#define REGISTER_SYCL_LZ(prim)    \
    struct attach_##prim##_impl { \
        attach_##prim##_impl();   \
    }

REGISTER_SYCL_LZ(rope);

#undef REGISTER_SYCL_LZ

}  // namespace detail
}  // namespace sycl_lz
}  // namespace cldnn
