// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace cldnn {
namespace sycl_lz {
#define ENABLE_MY_LOG 0
#if ENABLE_MY_LOG
#    ifndef DEBUG_PRINT
#        define DEBUG_PRINT(X) std::cout << "== " << __FILE__ << ":" << __LINE__ << " : " << X << std::endl
#        define DEBUG_PRINT_VAR(X) \
            std::cout << "== " << __FILE__ << ":" << __LINE__ << " : " << #X << " = " << X << std::endl
#    endif
#else
#    ifndef DEBUG_PRINT
#        define DEBUG_PRINT(X)
#        define DEBUG_PRINT_VAR(X)
#    endif
#endif

}  // namespace sycl_lz
}  // namespace cldnn
