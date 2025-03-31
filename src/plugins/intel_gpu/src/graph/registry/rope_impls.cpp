// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "registry.hpp"
#include "intel_gpu/primitives/rope.hpp"


// #if OV_GPU_WITH_ONEDNN
//     #include "impls/onednn/fully_connected_onednn.hpp"
// #endif

#if OV_GPU_WITH_SYCL_LZ
#    include "impls/sycl_lz/rope_sycl_lz.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<rope>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_SYCL_LZ(sycl_lz::RoPEImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_SYCL_LZ(sycl_lz::RoPEImplementationManager, shape_types::dynamic_shape)
        OV_GPU_GET_INSTANCE_OCL(rope, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(rope, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
