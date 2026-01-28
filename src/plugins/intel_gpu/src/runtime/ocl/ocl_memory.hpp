// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "ocl_engine.hpp"
#include "ocl_stream.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include <cassert>
#include <iterator>
#include <mutex>
#include <memory>

namespace cldnn {
namespace ocl {
class ocl_engine;

struct lockable_gpu_mem {
    lockable_gpu_mem() :
        _lock_count(0),
        _mapped_ptr(nullptr) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
};

struct gpu_buffer : public lockable_gpu_mem, public memory {
    gpu_buffer(ocl_engine* engine, const layout& new_layout, const cl::Buffer& buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_buffer(ocl_engine* engine, const layout& layout);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    event::ptr fill(stream& stream, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;
    const cl::Buffer& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }
    void* buffer_ptr() const override {
        return get_buffer().get();
    }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */, int64_t offset = 0) const override;
#endif

protected:
    cl::Buffer _buffer;
};

struct gpu_image2d : public lockable_gpu_mem, public memory {
    gpu_image2d(ocl_engine* engine, const layout& new_layout, const cl::Image2D& buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_image2d(ocl_engine* engine, const layout& layout);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    event::ptr fill(stream& stream, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;
    const cl::Image2D& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0, bool blocking = true) const override;

protected:
    cl::Image2D _buffer;
    size_t _width;
    size_t _height;
    size_t _row_pitch;
    size_t _slice_pitch;
};

struct gpu_media_buffer : public gpu_image2d {
    gpu_media_buffer(ocl_engine* engine, const layout& new_layout, shared_mem_params params);
    shared_mem_params get_internal_params() const override;
private:
    void* device;
#ifdef _WIN32
    void* surface;
#else
    uint32_t surface;
#endif
    uint32_t plane;
};

#ifdef _WIN32
struct gpu_dx_buffer : public gpu_buffer {
    gpu_dx_buffer(ocl_engine* engine, const layout& new_layout, shared_mem_params VAEncMiscParameterTypeSubMbPartPel);
    shared_mem_params get_internal_params() const override;
private:
    void* device;
    void* resource;
};
#endif

struct gpu_usm : public lockable_gpu_mem, public memory {
    gpu_usm(ocl_engine* engine, const layout& new_layout, const cl::UsmMemory& usm_buffer, allocation_type type, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(ocl_engine* engine, const layout& new_layout, const cl::UsmMemory& usm_buffer, std::shared_ptr<MemoryTracker> mem_tracker);
    gpu_usm(ocl_engine* engine, const layout& layout, allocation_type type);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    const cl::UsmMemory& get_buffer() const { return _buffer; }
    cl::UsmMemory& get_buffer() { return _buffer; }
    void* buffer_ptr() const override { return _buffer.get(); }

    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    event::ptr fill(stream& stream, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params() const override;

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */, int64_t offset = 0) const override;
#endif

    static allocation_type detect_allocation_type(const ocl_engine* engine, const void* mem_ptr);

    void release_usm_memory(std::function<void(const void*, size_t)> write_fn) {
        size_t bytes_count = 0;
        _buffer.getMemSize(bytes_count);

        if (bytes_count == 0) {
            OPENVINO_ASSERT(false, "[GPU] Weights size should not be zero!");
        } else {
            auto* ocl_engine = dynamic_cast<cldnn::ocl::ocl_engine*>(_engine);
            OPENVINO_ASSERT(ocl_engine != nullptr, "[GPU] OCL engine is not available for USM release");

            auto host_mem = ocl_engine->allocate_memory(_layout, allocation_type::usm_host, false);
            OPENVINO_ASSERT(host_mem != nullptr, "[GPU] Can't allocate host memory for USM release");

            auto host_usm = std::dynamic_pointer_cast<gpu_usm>(host_mem);
            OPENVINO_ASSERT(host_usm != nullptr, "[GPU] Host memory is not USM for USM release");

            auto& stream = ocl_engine->get_service_stream();
            host_usm->copy_from(stream, *this, 0, 0, bytes_count, true);

            write_fn(host_usm->buffer_ptr(), bytes_count);
        }

        _buffer.freeMem();
    }

    void load_usm_memory(std::function<size_t()> get_weights_size, std::function<void(const void*, size_t)> read_weights) {
        size_t bytes_count = get_weights_size();

        if (bytes_count == 0) {
            OPENVINO_ASSERT(false, "[GPU] Weights size should not be zero!");
        } else {
            OPENVINO_ASSERT(_buffer.get() == nullptr, "[GPU] USM buffer is already allocated!");
            _buffer.allocateDevice(bytes_count, nullptr);

            auto* ocl_engine = dynamic_cast<cldnn::ocl::ocl_engine*>(_engine);
            OPENVINO_ASSERT(ocl_engine != nullptr, "[GPU] OCL engine is not available for USM release");

            auto host_mem = ocl_engine->allocate_memory(_layout, allocation_type::usm_host, false);
            OPENVINO_ASSERT(host_mem != nullptr, "[GPU] Can't allocate host memory for USM release");

            auto host_usm = std::dynamic_pointer_cast<gpu_usm>(host_mem);
            OPENVINO_ASSERT(host_usm != nullptr, "[GPU] Host memory is not USM for USM release");

            auto& stream = ocl_engine->get_service_stream();
            read_weights(host_usm->buffer_ptr(), bytes_count);

            this->copy_from(stream, *host_usm, 0, 0, bytes_count, true);
            host_usm->get_buffer().freeMem();
        }
    }

protected:
    cl::UsmMemory _buffer;
    cl::UsmMemory _host_buffer;

    static allocation_type detect_allocation_type(const ocl_engine* engine, const cl::UsmMemory& buffer);
};

struct ocl_surfaces_lock : public surfaces_lock {
    ocl_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream);

    ~ocl_surfaces_lock() = default;
private:
    std::vector<cl_mem> get_handles(std::vector<memory::ptr> mem) const;
    std::vector<cl_mem> _handles;
    std::unique_ptr<cl::SharedSurfLock> _lock;
};
}  // namespace ocl
}  // namespace cldnn
