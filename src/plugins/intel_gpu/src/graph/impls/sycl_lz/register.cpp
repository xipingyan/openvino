// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "register.hpp"

namespace cldnn {
namespace sycl_lz {

#define REGISTER_SYCL_LZ(prim) static detail::attach_##prim##_impl attach_##prim

void register_implementations() {
    REGISTER_SYCL_LZ(rope);
}

}  // namespace sycl_lz
}  // namespace cldnn
