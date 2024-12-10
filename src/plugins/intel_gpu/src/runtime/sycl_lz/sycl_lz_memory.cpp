// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "sycl_lz_memory.hpp"
#include "sycl_lz_engine.hpp"
#include "sycl_lz_stream.hpp"
// #include "sycl_lz_event.hpp"
#include <stdexcept>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ocl.hpp>
#endif

#define TRY_CATCH_CL_ERROR(...)               \
    try {                                     \
        __VA_ARGS__;                          \
    } catch (cl::Error const& err) {          \
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err)); \
    }

namespace cldnn {
namespace sycl_lz {

gpu_usm::gpu_usm(sycl_lz_engine* engine, const layout& layout, allocation_type type)
    : lockable_gpu_mem(),
      memory(engine, layout, type, nullptr),
      _buffer(engine->get_usm_helper()) {
    DEBUG_PRINT_VAR(layout);
    GPU_DEBUG_LOG << "gpu_usm, type=" << type << ", layout.get_shape()" << layout.get_shape() << std::endl;

    // , _host_buffer(engine->get_usm_helper()) {

    switch (get_allocation_type()) {
    case allocation_type::usm_host:
        _buffer.allocateHost(_bytes_count);
        break;
    case allocation_type::usm_shared:
        _buffer.allocateShared(_bytes_count);
        break;
    case allocation_type::usm_device:
        _buffer.allocateDevice(_bytes_count);
        break;
    default:
        CLDNN_ERROR_MESSAGE("gpu_usm allocation type",
            "Unknown unified shared memory type!");
    }

    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _buffer.get(), layout.bytes_count(), type);
}

void* gpu_usm::lock(const stream& stream, mem_lock_type type) {
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        // auto& sycllz_stream = downcast<const sycl_lz_stream>(stream);
        if (get_allocation_type() == allocation_type::usm_device) {
            if (type != mem_lock_type::read) {
                throw std::runtime_error("Unable to lock allocation_type::usm_device with write lock_type.");
            }
            GPU_DEBUG_LOG << "Copy usm_device buffer to host buffer." << std::endl;

            DEBUG_PRINT("Not implemented. _host_buffer.allocateHost(_bytes_count);");
            // _host_buffer.allocateHost(_bytes_count);
            // try {
            //     sycl_lz_stream.get_usm_helper().enqueue_memcpy(sycllz_stream.get_cl_queue(), _host_buffer.get(), _buffer.get(), _bytes_count, CL_TRUE);
            // } catch (cl::Error const& err) {
            //     OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
            // }
            // _mapped_ptr = _host_buffer.get();
        } else {
            _mapped_ptr = _buffer.get();
        }
    }
    _lock_count++;
    return _mapped_ptr;
}

void gpu_usm::unlock(const stream& /* stream */) {
    std::lock_guard<std::mutex> locker(_mutex);

    _lock_count--;
    if (0 == _lock_count) {
        if (get_allocation_type() == allocation_type::usm_device) {
            DEBUG_PRINT("Not implemented.");
            // _host_buffer.freeMem();
        }
        _mapped_ptr = nullptr;
    }
}

event::ptr gpu_usm::fill(stream& stream, unsigned char pattern, bool blocking) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;

    // if (_bytes_count == 0) {
    //     GPU_DEBUG_TRACE_DETAIL << "Skip gpu_usm::fill for 0 size tensor" << std::endl;
    //     return nullptr;
    // }
    // auto& cl_stream = downcast<sycl_lz_stream>(stream);
    // auto ev = stream.create_base_event();
    // cl::Event& ev_ocl = downcast<sycl_lz_event>(ev.get())->get();
    // try {
    //     cl_stream.get_usm_helper().enqueue_fill_mem(
    //             cl_stream.get_cl_queue(), _buffer.get(), static_cast<const void*>(&pattern), sizeof(unsigned char), _bytes_count, nullptr, &ev_ocl);
    //     if (blocking) {
    //         ev_ocl.wait();
    //     }
    // } catch (cl::Error const& err) {
    //     OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    // }

    // return ev;
}

event::ptr gpu_usm::fill(stream& stream, bool blocking) {
    return fill(stream, 0, blocking);
}

event::ptr gpu_usm::copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;

    // auto result_event = create_event(stream, size, blocking);
    // if (size == 0)
    //     return result_event;

    // auto cl_stream = downcast<ocl_stream>(&stream);
    // auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();
    // auto src_ptr = reinterpret_cast<const char*>(data_ptr) + src_offset;
    // auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;

    // TRY_CATCH_CL_ERROR(cl_stream->get_usm_helper().enqueue_memcpy(cl_stream->get_cl_queue(), dst_ptr, src_ptr, size, blocking, nullptr, cl_event));

    // return result_event;
}

event::ptr gpu_usm::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;

    // auto result_event = create_event(stream, size, blocking);
    // if (size == 0)
    //     return result_event;

    // auto cl_stream = downcast<ocl_stream>(&stream);
    // auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();

    // if (src_mem.get_allocation_type() == allocation_type::cl_mem) {
    //     auto cl_mem_buffer = downcast<const gpu_buffer>(&src_mem);
    //     auto dst_ptr = reinterpret_cast<char*>(buffer_ptr());

    //     return cl_mem_buffer->copy_to(stream, dst_ptr, src_offset, dst_offset, size, blocking);
    // } else if (memory_capabilities::is_usm_type(src_mem.get_allocation_type())) {
    //     auto usm_mem = downcast<const gpu_usm>(&src_mem);
    //     auto src_ptr = reinterpret_cast<const char*>(usm_mem->buffer_ptr()) + src_offset;
    //     auto dst_ptr = reinterpret_cast<char*>(buffer_ptr()) + dst_offset;

    //     TRY_CATCH_CL_ERROR(cl_stream->get_usm_helper().enqueue_memcpy(cl_stream->get_cl_queue(), dst_ptr, src_ptr, size, blocking, nullptr, cl_event));
    // } else {
    //     std::vector<char> tmp_buf;
    //     tmp_buf.resize(size);
    //     src_mem.copy_to(stream, tmp_buf.data(), src_offset, 0, size, true);

    //     GPU_DEBUG_TRACE_DETAIL << "Suboptimal copy call from " << src_mem.get_allocation_type() << " to " << get_allocation_type() << "\n";
    //     return copy_from(stream, tmp_buf.data(), 0, 0, size, blocking);
    // }

    // return result_event;
}

event::ptr gpu_usm::copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
    // auto result_event = create_event(stream, size, blocking);
    // if (size == 0)
    //     return result_event;

    // auto cl_stream = downcast<ocl_stream>(&stream);
    // auto cl_event = blocking ? nullptr : &downcast<ocl_event>(result_event.get())->get();
    // auto src_ptr = reinterpret_cast<const char*>(buffer_ptr()) + src_offset;
    // auto dst_ptr = reinterpret_cast<char*>(data_ptr) + dst_offset;

    // TRY_CATCH_CL_ERROR(cl_stream->get_usm_helper().enqueue_memcpy(cl_stream->get_cl_queue(), dst_ptr, src_ptr, size, blocking, nullptr, cl_event));

    // return result_event;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::memory gpu_usm::get_onednn_memory(dnnl::memory::desc desc, int64_t offset) const {
    auto onednn_engine = _engine->get_onednn_engine();
    dnnl::memory dnnl_mem = dnnl::ocl_interop::make_memory(desc, onednn_engine, dnnl::ocl_interop::memory_kind::usm,
        reinterpret_cast<uint8_t*>(_buffer.get()) + offset);
    return dnnl_mem;
}
#endif

shared_mem_params gpu_usm::get_internal_params() const {
    auto cl_engine = downcast<const sycl_lz_engine>(_engine);
//     return {
//         shared_mem_type::shared_mem_usm,  // shared_mem_type
//         static_cast<shared_handle>(cl_engine->get_cl_context().get()),  // context handle
//         nullptr,        // user_device handle
//         _buffer.get(),  // mem handle
// #ifdef _WIN32
//         nullptr,  // surface handle
// #else
//         0,  // surface handle
// #endif
//         0  // plane
//     };

    return {
        shared_mem_type::shared_mem_usm,  // shared_mem_type
        static_cast<shared_handle>(nullptr),  // context handle
        nullptr,        // user_device handle
        _buffer.get(),  // mem handle
#ifdef _WIN32
        nullptr,  // surface handle
#else
        0,  // surface handle
#endif
        0  // plane
    };
}

allocation_type gpu_usm::detect_allocation_type(const sycl_lz_engine* engine, const void* mem_ptr) {
    DEBUG_PRINT("Not implemented.");
    // auto cl_alloc_type = engine->get_usm_helper().get_usm_allocation_type(mem_ptr);

    // allocation_type res;
    // switch (cl_alloc_type) {
    //     case CL_MEM_TYPE_DEVICE_INTEL: res = allocation_type::usm_device; break;
    //     case CL_MEM_TYPE_HOST_INTEL: res = allocation_type::usm_host; break;
    //     case CL_MEM_TYPE_SHARED_INTEL: res = allocation_type::usm_shared; break;
    //     default: res = allocation_type::unknown;
    // }

    return allocation_type::usm_shared;
}

allocation_type gpu_usm::detect_allocation_type(const sycl_lz_engine* engine, const void* & buffer) {
    DEBUG_PRINT("Not implemented.");
    return allocation_type::usm_shared;
    // auto alloc_type = detect_allocation_type(engine, buffer.get());
    // OPENVINO_ASSERT(alloc_type == allocation_type::usm_device ||
    //                 alloc_type == allocation_type::usm_host ||
    //                 alloc_type == allocation_type::usm_shared, "[GPU] Unsupported USM alloc type: " + to_string(alloc_type));
    // return alloc_type;
}

}  // namespace sycl_lz
}  // namespace cldnn
