// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive_inst.h"
#include "intel_gpu/runtime/memory.hpp"
#include "impls/registry/registry.hpp"

#include <vector>

#include "sycl/sycl.hpp"

namespace cldnn {
namespace sycl_lz {

static std::mutex cacheAccessMutex;

template <class PType>
struct typed_primitive_sycl_lz_impl : public typed_primitive_impl<PType> {
    const engine* _engine;

    typed_primitive_sycl_lz_impl(const engine& engine,
                                 const ExecutionConfig& config,
                                 std::shared_ptr<WeightsReorderParams> weights_reorder = nullptr)
        : typed_primitive_impl<PType>(weights_reorder, "sycl_lz_kernel"),
          _engine(&engine) {
        GPU_DEBUG_LOG << "== typed_primitive_sycl_lz_impl sycl_lz_kernel base class." << std::endl;
    }

    typed_primitive_sycl_lz_impl() : typed_primitive_impl<PType>({}, "undef"), _engine(nullptr) {
        GPU_DEBUG_LOG << "== typed_primitive_sycl_lz_impl undef base class." << std::endl;
    }

    bool is_cpu() const override { return false; }
    bool is_onednn() const override { return false; }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override { }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override { }
    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override { }

    std::vector<layout> get_internal_buffer_layouts_impl() const override {
        GPU_DEBUG_LOG << "== Not Implemented. get_internal_buffer_layouts_impl." << std::endl;
        return {};
    }
};

}  // namespace sycl_lz
}  // namespace cldnn
