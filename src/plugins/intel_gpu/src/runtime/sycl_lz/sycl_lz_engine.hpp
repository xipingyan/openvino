// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "sycl/sycl.hpp"
#include "sycl_lz_device.hpp"
#include "sycl_lz_ext.hpp"

namespace cldnn {
namespace sycl_lz {

class sycl_lz_engine : public engine {
public:
    sycl_lz_engine(const device::ptr dev, runtime_types runtime_type);

    stream_ptr create_stream(const ExecutionConfig& config) const override;
    stream_ptr create_stream(const ExecutionConfig& config, void *handle) const override;

    engine_types type() const override { return engine_types::sycl_lz; };
    runtime_types runtime_type() const override { return runtime_types::sycl_lz; };

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;
    bool check_allocatable(const layout& layout, allocation_type type) override;

    void* get_user_context() const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::usm_shared; }
    allocation_type detect_usm_allocation_type(const void* memory) const override;

    bool extension_supported(std::string extension) const;

    stream& get_service_stream() const override;

    kernel::ptr prepare_kernel(const kernel::ptr kernel) const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    void create_onednn_engine(const ExecutionConfig& config) override;
    // Returns onednn engine object which shares device and context with current engine
    dnnl::engine& get_onednn_engine() const override;
#endif

    ::sycl::context& get_sycl_context() const;
    std::unique_ptr<::sycl::context> sycl_context = nullptr;
    const ::sycl_lz::UsmHelper& get_usm_helper() const;

private:
    std::unique_ptr<sycl::queue> _queue = nullptr;

    std::string _extensions;
    std::unique_ptr<stream> _service_stream;
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::mutex onednn_mutex;
    std::shared_ptr<dnnl::engine> _onednn_engine;
#endif
};

}  // namespace sycl_lz
}  // namespace cldnn
