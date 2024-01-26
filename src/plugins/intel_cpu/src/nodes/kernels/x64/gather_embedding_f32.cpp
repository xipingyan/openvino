// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_embedding_f32.hpp"

#include "openvino/core/except.hpp"
#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif
#include "openvino/core/parallel.hpp"

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
                               const bool& reverseIndexing) {
    const auto& idxDims = idxMemPtr->getStaticDims();
    const auto batch = (idxDims.size() == 0) ? 1 : idxDims[0];
    const auto seqLen = (idxDims.size() == 0) ? 1 : idxDims[1];

    auto axisDim = srcMemPtr->getStaticDims()[0];
    auto feaDim = srcMemPtr->getStaticDims()[1];
    static const auto* use_gopt = std::getenv("USE_GOPT");
#if 0  // parallel_nt
    size_t core_num = 24;
    float step = static_cast<float>(feaDim) / core_num;
    parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
        size_t start = 0, end = 0;
        splitter(core_num, nthr, ithr, start, end);
        if (start == end) {
            return;
        }
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

                size_t rS = start * step;
                size_t rE = end * step;
                // printf("%d, %d, start=%d, end=%d, [%lu, %lu]\n", nthr, ithr, start, end, rS, rE);
                for (size_t f = rS; f < rE; f++) {
                    pdst[dstIdx * feaDim + f] = (static_cast<float>(psrc[ii * feaDim + f]) - zp[ii]) * scale[ii];
                }
            }
        }
    });
    return;
#endif

#if 1  // parallel_for2d in batch and seq
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
        float* pdst_cur = pdst + dstIdx * feaDim;
        size_t f = 0;
#    if __AVX2__
        if (use_gopt) {
            __m256 mm_zp = _mm256_set1_ps(zp[ii]);
            __m256 mm_scale = _mm256_set1_ps(scale[ii]);
            for (; f + 8 <= feaDim; f += 8) {
                auto x0 = _mm_loadu_si64(psrc_cur + f); // 6 0.333
                auto y0 = _mm256_cvtepu8_epi32(x0);     // 3   1
                auto z0 = _mm256_cvtepi32_ps(y0);       // 4 0.5
                z0 = _mm256_sub_ps(z0, mm_zp);          // 2 0.5
                z0 = _mm256_mul_ps(z0, mm_scale);       // 4 0.5
                _mm256_storeu_ps(pdst_cur + f, z0);         // 1 0.5
                //asm("int3");
            }
        }
#    endif
        for (; f < feaDim; f++) {
            pdst_cur[f] = (static_cast<float>(psrc_cur[f]) - zp[ii]) * scale[ii];
        }
    });
    return;
#endif

#if 0
    // Reference implementation: Merge batch and seq
    for (size_t i = 0; i < batch * seqLen; i++) {
        auto ii = pidx[i];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        auto srcIdx = ii * feaDim;

        size_t f = 0;
        const uint8_t* psrc_cur = psrc + srcIdx;
#    if __AVX2__
        if (use_gopt) {
            __m256 mm_zp = _mm256_set1_ps(zp[ii]);
            __m256 mm_scale = _mm256_set1_ps(scale[ii]);
            for (; f + 8 <= feaDim; f += 8) {
                auto x0 = _mm_loadu_si64(psrc_cur + f); // 6 0.333
                auto y0 = _mm256_cvtepu8_epi32(x0);     // 3   1
                auto z0 = _mm256_cvtepi32_ps(y0);       // 4 0.5
                z0 = _mm256_sub_ps(z0, mm_zp);          // 2 0.5
                z0 = _mm256_mul_ps(z0, mm_scale);       // 4 0.5
                _mm256_storeu_ps(pdst + f, z0);         // 1 0.5
                //asm("int3");
            }
            pdst += feaDim;
            continue;
        }
#    endif
        for (; f < feaDim; f++) {
            pdst[f] = (static_cast<float>(psrc_cur[f]) - zp[ii]) * scale[ii];
        }
        pdst += feaDim;
    }
    return;
#endif
    // Reference implementation: Original
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
                pdst[dstIdx * feaDim + f] = (static_cast<float>(psrc[ii * feaDim + f]) - zp[ii]) * scale[ii];
                asm("int3");
            }
        }
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov