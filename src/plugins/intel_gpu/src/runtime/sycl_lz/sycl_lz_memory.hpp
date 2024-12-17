// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_lz_common.hpp"
#include "sycl_lz_engine.hpp"
#include "sycl_lz_stream.hpp"
#include "sycl_lz_ext.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include <cassert>
#include <iterator>
#include <mutex>
#include <memory>

namespace cldnn {
namespace sycl_lz {
struct lockable_gpu_mem {
    lockable_gpu_mem() :
        _lock_count(0),
        _mapped_ptr(nullptr) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
};

// "Not implemented[SYCL_RUNTIME]. "
// Only implement gpu_usm buffer currently.

struct gpu_usm : public lockable_gpu_mem, public memory {
    gpu_usm(sycl_lz_engine* engine, const layout& new_layout, void*& usm_buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(sycl_lz_engine* engine, const layout& new_layout, void*& usm_buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(sycl_lz_engine* engine, const layout& layout, allocation_type type);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    const ::sycl_lz::UsmMemory & get_buffer() const { return _buffer; }
    ::sycl_lz::UsmMemory& get_buffer() { return _buffer; }
    void* buffer_ptr() const override { return _buffer.get(); }

    event::ptr fill(stream& stream, unsigned char pattern, bool blocking = true) override;
    event::ptr fill(stream& stream, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */, int64_t offset = 0) const override;
#endif

    static allocation_type detect_allocation_type(const sycl_lz_engine* engine, const void* mem_ptr);

protected:
    // void* _buffer;
    // void* _host_buffer;
    ::sycl_lz::UsmMemory _buffer;
    // ::sycl_lz::UsmMemory _host_buffer;

    static allocation_type detect_allocation_type(const sycl_lz_engine* engine, const void*& buffer);
};

struct sycl_lz_surfaces_lock : public surfaces_lock {
    sycl_lz_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream) {
        GPU_DEBUG_LOG << "Not implemented[SYCL_RUNTIME]. " << std::endl;
    }

    ~sycl_lz_surfaces_lock() = default;
private:
    std::vector<void*> get_handles(std::vector<memory::ptr> mem) const {
        return _handles;
    }
    std::vector<void*> _handles;
    // std::unique_ptr<cl::SharedSurfLock> _lock;
};

}  // namespace sycl_lz
}  // namespace cldnn
