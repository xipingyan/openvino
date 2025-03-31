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
    static std::unique_ptr<primitive_impl> create(const typed_program_node<PType>& arg, const kernel_impl_params& impl_param) {
        // // concat buffer fusing for dynamic shape is adaptively applied at runtime. So we need to build dynamic impl at build time.
        // if (impl_param.can_be_optimized() &&
        //     !((impl_param.is_type<concatenation>() || impl_param.is_type<crop>() || impl_param.runtime_skippable()) && impl_param.is_dynamic())) {
        //     return std::make_unique<ImplType>(kernel_selector::kernel_data{});
        // }
        // auto kernel_params = ImplType::get_kernel_params(ImplType::static_canonicalize_shapes(impl_param));
        // kernel_params.is_shape_agnostic = impl_param.is_dynamic();
        // kernel_params.set_dynamic_shape_offsets();
        // auto& kernel_selector = ImplType::kernel_selector_t::Instance();
        // auto best_kernel = kernel_selector.get_best_kernel(kernel_params);

        // return std::make_unique<ImplType>(best_kernel);
        return nullptr;
    }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override { }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override { }
    void set_arguments_impl(typed_primitive_inst<PType>& instance, kernel_arguments_data& args) override { }

    template <typename ImplType, typename KernelParamsType>
    static std::unique_ptr<primitive_impl> make_deep_copy(const ImplType& impl_sycl) {
        auto prim_impl = std::make_unique<ImplType>(impl_sycl);
        KernelParamsType* params_ptr = dynamic_cast<KernelParamsType*>((*prim_impl)._kernel_data.params.get());
        if (params_ptr != nullptr) {
            (*prim_impl)._kernel_data.params = std::make_unique<KernelParamsType>(*params_ptr);
        }
        return prim_impl;
    }
};

}  // namespace sycl_lz
}  // namespace cldnn
