// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// \file Just keep algin with ocl_ext.hpp
///

#pragma once

#include <array>
#include <sycl/sycl.hpp>

#include "intel_gpu/runtime/debug_configuration.hpp"

namespace sycl_lz {

class UsmHelper {
public:
    explicit UsmHelper(const sycl::context& ctx, const sycl::device device, bool use_usm) : _ctx(ctx), _device(device) {
    }

    void* allocate_host(const cl_mem_properties_intel* properties,
                        size_t size,
                        cl_uint alignment,
                        cl_int* err_code_ret) const {
        GPU_DEBUG_LOG << "sycl::malloc_host(" << size << ")" << std::endl;
        return sycl::malloc_host(size, _ctx);
    }

    void* allocate_shared(const cl_mem_properties_intel* properties,
                          size_t size,
                          cl_uint alignment,
                          cl_int* err_code_ret) const {
        GPU_DEBUG_LOG << "sycl::malloc_shared(" << size << ")" << std::endl;
        return sycl::malloc_shared(size, _device, _ctx);
    }

    void* allocate_device(const cl_mem_properties_intel* properties,
                          size_t size,
                          cl_uint alignment,
                          cl_int* err_code_ret) const {
        GPU_DEBUG_LOG << "sycl::malloc_device(" << size << ")" << std::endl;
        return sycl::malloc_device(size, _device, _ctx);
    }

    void free_mem(void* ptr) const {
        sycl::free(ptr, _ctx);
    }

    cl_int set_kernel_arg_mem_pointer(const sycl::kernel& kernel, uint32_t index, const void* ptr) const {
        GPU_DEBUG_LOG << "Not implemented." << std::endl;
        return 0;
    }

    cl_int enqueue_memcpy(const sycl::queue& cpp_queue,
                          void* dst_ptr,
                          const void* src_ptr,
                          size_t bytes_count,
                          bool blocking = true,
                          const std::vector<sycl::event>* wait_list = nullptr,
                          sycl::event* ret_event = nullptr) const {
        GPU_DEBUG_LOG << "Temp Solution, Copy device buffer to host. " << std::endl;

        // cpp_queue.copy<void*>(src_ptr, dst_ptr, bytes_count);
        auto new_queue = cpp_queue;
        auto ev = new_queue.submit([&](sycl::handler& cgh) {
            cgh.memcpy(dst_ptr, src_ptr, bytes_count);
        });

        if (ret_event) {
            // return sycl_stream->create_base_event(result_event);
            *ret_event = ev;
        } else {
            ev.wait();
            // sycl::half* dst_ptr_f16 = static_cast<sycl::half*>(dst_ptr);
            // GPU_DEBUG_LOG << "  == host dst_ptr[0:3]=" << static_cast<float>(dst_ptr_f16[0]) << ", "
            //               << static_cast<float>(dst_ptr_f16[1]) << ", " << static_cast<float>(dst_ptr_f16[2]) << ", "
            //               << static_cast<float>(dst_ptr_f16[3]) << std::endl;
        }
        return 0;
    }

    cl_int enqueue_fill_mem(const sycl::queue& cpp_queue,
                            void* dst_ptr,
                            const void* pattern,
                            size_t pattern_size,
                            size_t bytes_count,
                            const std::vector<sycl::event>* wait_list = nullptr,
                            sycl::event* ret_event = nullptr) const {
        auto new_queue = cpp_queue;
        OPENVINO_ASSERT(pattern_size == 1u, "Only support: pattern_size == 1.");
        OPENVINO_ASSERT(wait_list == nullptr, "Only support: wait_list == nullptr");
        const char pattern_value = *reinterpret_cast<const char*>(pattern);

        GPU_DEBUG_LOG << "Temp Solution, replace fill with memcpy, fill memory: pattern=" << pattern_value << ", pattern_size=" << pattern_size
                      << ", bytes_count=" << bytes_count << std::endl;

        /* Next codes will trigger exception: "No kernel named  was found".
            I don't why my unit test work for next codes. Strange!!!
         */
        // auto ev = new_queue.fill(dst_ptr, pattern_value, bytes_count);
        // auto ev = new_queue.submit([&](sycl::handler& cgh) {
        //     cgh.fill(dst_ptr, pattern_value, bytes_count);
        // });

        // Replace fill with memcpy. Temp solution.
        void* tmp_buf = malloc(bytes_count);
        memset(tmp_buf, 0, bytes_count);
        new_queue
            .submit([&](sycl::handler& cgh) {
                cgh.memcpy(dst_ptr, tmp_buf, bytes_count);
            })
            .wait();
        free(tmp_buf);
        sycl::event ev;

        if (ret_event) {
            *ret_event = ev;
        } else {
            ev.wait();
        }
        return 0;
    }

    cl_int enqueue_set_mem(const sycl::queue& cpp_queue,
                           void* dst_ptr,
                           cl_int value,
                           size_t bytes_count,
                           const std::vector<sycl::event>* wait_list = nullptr,
                           sycl::event* ret_event = nullptr) const {
        GPU_DEBUG_LOG << "Not implemented." << std::endl;
        return 0;
    }

    sycl::usm::alloc get_usm_allocation_type(const void* usm_ptr) const {
        return sycl::get_pointer_type(usm_ptr, _ctx);
    }

    size_t get_usm_allocation_size(const void* usm_ptr) const {
        GPU_DEBUG_LOG << "Not implemented." << std::endl;
        return 0;
    }

private:
    sycl::context _ctx;
    sycl::device _device;
};

/*
UsmPointer requires associated context to free it.
Simple wrapper class for usm allocated pointer.
*/
class UsmHolder {
public:
    UsmHolder(const sycl_lz::UsmHelper& usmHelper, void* ptr, bool shared_memory = false)
        : _usmHelper(usmHelper),
          _ptr(ptr),
          _shared_memory(shared_memory) {}

    void* ptr() {
        return _ptr;
    }
    void memFree() {
        try {
            if (!_shared_memory)
                _usmHelper.free_mem(_ptr);
        } catch (...) {
            // Exception may happen only when clMemFreeINTEL function is unavailable, thus can't free memory properly
        }
        _ptr = nullptr;
    }
    ~UsmHolder() {
        memFree();
    }

private:
    const sycl_lz::UsmHelper& _usmHelper;
    void* _ptr;
    bool _shared_memory = false;
};

class UsmMemory {
public:
    explicit UsmMemory(const sycl_lz::UsmHelper& usmHelper) : _usmHelper(usmHelper) {}
    UsmMemory(const sycl_lz::UsmHelper& usmHelper, void* usm_ptr, size_t offset = 0)
        : _usmHelper(usmHelper)
        , _usm_pointer(std::make_shared<UsmHolder>(_usmHelper, reinterpret_cast<uint8_t*>(usm_ptr) + offset, true)) {
        if (!usm_ptr) {
            throw std::runtime_error("[GPU] Can't share null usm pointer");
        }
    }

    // Get methods returns original pointer allocated by openCL.
    void* get() const {
        return _usm_pointer->ptr();
    }

    void allocateHost(size_t size) {
        cl_int error = CL_SUCCESS;
        auto ptr = _usmHelper.allocate_host(nullptr, size, 0, &error);
        _check_error(size, ptr, error, "Host");
        _allocate(ptr);
    }

    void allocateShared(size_t size) {
        cl_int error = CL_SUCCESS;
        auto ptr = _usmHelper.allocate_shared(nullptr, size, 0, &error);
        _check_error(size, ptr, error, "Shared");
        _allocate(ptr);
    }

    void allocateDevice(size_t size) {
        cl_int error = CL_SUCCESS;
        auto ptr = _usmHelper.allocate_device(nullptr, size, 0, &error);
        _check_error(size, ptr, error, "Device");
        _allocate(ptr);
    }

    void freeMem() {
        if (!_usm_pointer)
            throw std::runtime_error("[CL ext] Can not free memory of empty UsmHolder");
        _usm_pointer->memFree();
    }

    virtual ~UsmMemory() = default;

protected:
    const UsmHelper& _usmHelper;
    std::shared_ptr<UsmHolder> _usm_pointer = nullptr;

private:
    void _allocate(void* ptr) {
        _usm_pointer = std::make_shared<UsmHolder>(_usmHelper, ptr);
    }

    void _check_error(size_t size, void* ptr, cl_int error, const char* usm_type) {
        // if (ptr == nullptr || error != CL_SUCCESS) {
        //     std::stringstream sout;
        //     sout << "[CL ext] Can not allocate " << size << " bytes for USM " << usm_type << ". ptr: " << ptr
        //          << ", error: " << error << std::endl;
        //     if (ptr == nullptr)
        //         throw std::runtime_error(sout.str());
        //     else
        //         detail::errHandler(error, sout.str().c_str());
        // }
    }
};

/*
    Wrapper for standard cl::Kernel object.
    Extend cl::Kernel functionality.
*/
// class SyclLzKernelIntel : public Kernel {
class SyclLzKernelIntel {
public:
    // explicit SyclLzKernelIntel(const UsmHelper& usmHelper) : _usmHelper(usmHelper) {}
    SyclLzKernelIntel(const sycl::kernel& other, const UsmHelper& usmHelper)
        : _kernel(other),
          _usmHelper(usmHelper) {}

    SyclLzKernelIntel clone() const {
        return SyclLzKernelIntel(_kernel, this->_usmHelper);
    }

    cl_int setArgUsm(cl_uint index, const UsmMemory& mem) {
        GPU_DEBUG_LOG << "Not implemented. Maybe not need for sycl kernel" << std::endl;
        // return detail::errHandler(_usmHelper.set_kernel_arg_mem_pointer(*this, index, mem.get()), "[CL_EXT] setArgUsm in KernelIntel failed");
        return 0;
    }

    sycl::kernel get_kernel() {
        return _kernel;
    }

private:
    sycl::kernel _kernel;
    const UsmHelper& _usmHelper;
};

inline bool operator==(const UsmMemory& lhs, const UsmMemory& rhs) {
    return lhs.get() == rhs.get();
}

inline bool operator!=(const UsmMemory& lhs, const UsmMemory& rhs) {
    return !operator==(lhs, rhs);
}
}  // namespace sycl_lz