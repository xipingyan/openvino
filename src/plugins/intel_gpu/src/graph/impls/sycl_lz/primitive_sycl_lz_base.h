// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive_inst.h"
#include "intel_gpu/runtime/memory.hpp"
#include "registry/registry.hpp"

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

    template <typename ImplType>
    static std::unique_ptr<ImplType> make_deep_copy(const ImplType* impl) {
        auto copy = std::make_unique<ImplType>();  // Use default c-tor to initialize stages
        copy->_order = impl->_order;
        copy->m_rt_params = nullptr;  // don't copy RT params
        copy->m_manager = impl->m_manager;
        copy->can_reuse_memory = impl->can_reuse_memory;
        copy->can_share_kernels = impl->can_share_kernels;
        copy->_weights_reorder_params = impl->_weights_reorder_params;
        copy->_kernel_name = impl->_kernel_name;
        copy->_is_dynamic = impl->_is_dynamic;

        for (size_t i = 0; i < copy->_stages.size(); i++) {
            copy->_stages[i]->kd = impl->_stages[i]->kd;
            if (impl->_stages[i]->kernel) {
                copy->_stages[i]->kernel = impl->_stages[i]->kernel->clone();
            }
        }

        return copy;
    }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override { }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override { }
    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override { }
};

}  // namespace sycl_lz
}  // namespace cldnn
