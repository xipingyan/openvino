// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_embedding_bf16.hpp"

#include "openvino/core/except.hpp"
#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif
#include "openvino/core/parallel.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void execInputEmbeddingCasebf16(ov::intel_cpu::MemoryPtr& srcMemPtr,
                                ov::intel_cpu::MemoryPtr& idxMemPtr,
                                const uint8_t* psrc,
                                const int32_t* pidx,
                                bfloat16* pdst,
                                const float_t* zp,
                                const float_t* scale,
                                const bool& reverseIndexing) {
    const auto& idxDims = idxMemPtr->getStaticDims();
    const auto batch = (idxDims.size() == 0) ? 1 : idxDims[0];
    const auto seqLen = (idxDims.size() == 0) ? 1 : idxDims[1];

    auto axisDim = srcMemPtr->getStaticDims()[0];
    auto feaDim = srcMemPtr->getStaticDims()[1];
    static const auto* use_gopt = std::getenv("USE_GOPT");

#if 1
    parallel_for2d(batch, seqLen, [&](const size_t b, const size_t s) {
        auto dstIdx = b * seqLen + s;
        auto ii = pidx[dstIdx];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        const uint8_t* psrc_cur = psrc + ii * feaDim;
        bfloat16* pdst_cur = pdst + dstIdx * feaDim;
        size_t f = 0;
#    if __AVX2__
        // if (use_gopt) {
        //     __m256 mm_zp = _mm256_set1_ps(zp[ii]);
        //     __m256 mm_scale = _mm256_set1_ps(scale[ii]);
        //     for (; f + 8 <= feaDim; f += 8) {
        //         auto x0 = _mm_loadu_si64(psrc_cur + f); // 6 0.333
        //         auto y0 = _mm256_cvtepu8_epi32(x0);     // 3   1
        //         auto z0 = _mm256_cvtepi32_ps(y0);       // 4 0.5
        //         z0 = _mm256_sub_ps(z0, mm_zp);          // 2 0.5
        //         z0 = _mm256_mul_ps(z0, mm_scale);       // 4 0.5
        //         // f32->bf16
        //         // ????
        //         // _mm256_storeu_ps(pdst_cur + f, z0);         // 1 0.5
        //         //asm("int3");
        //     }
        // }
#    endif
        for (; f < feaDim; f++) {
            pdst_cur[f] = static_cast<bfloat16>((static_cast<float>(psrc_cur[f]) - zp[ii]) * scale[ii]);
        }
    });
    return;
#endif

    // Reference implementation
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seqLen; s++) {
            auto dstIdx = b * seqLen + s;
            auto ii = pidx[dstIdx];
            if (ii < 0) {
                if (reverseIndexing)
                    ii += axisDim;
                else
                    ii = axisDim;
            }
            for (size_t f = 0; f < feaDim; f++) {
                pdst[dstIdx * feaDim + f] =
                    static_cast<bfloat16>((static_cast<float>(psrc[ii * feaDim + f]) - zp[ii]) * scale[ii]);
            }
        }
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov