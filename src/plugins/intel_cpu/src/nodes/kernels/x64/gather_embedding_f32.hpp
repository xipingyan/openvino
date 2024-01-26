// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_types.h"
#include "cpu_memory.h"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void execInputEmbeddingCasef32(ov::intel_cpu::MemoryPtr& srcMemPtr,
                               ov::intel_cpu::MemoryPtr& idxMemPtr,
                               const uint8_t* psrc,
                               const int32_t* pidx,
                               float* pdst,
                               const float_t* zp,
                               const float_t* scale,
                               const bool& reverseIndexing);
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov