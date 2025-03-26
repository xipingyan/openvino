// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_lz_engine.hpp"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "sycl_lz_memory.hpp"
#include "sycl_lz_stream.hpp"

namespace cldnn {
namespace sycl_lz {


sycl_lz_engine::sycl_lz_engine(const device::ptr dev, runtime_types runtime_type)
: engine(dev) {
    GPU_DEBUG_INFO << "create sycl_lz_engine" << std::endl;

    auto casted_dev = dynamic_cast<sycl_lz::sycl_lz_device*>(_device.get());
    auto device = casted_dev->get_device();
    sycl_context = std::make_unique<::sycl::context>(device);

    _service_stream.reset(new sycl_lz_stream(*this, ExecutionConfig()));
}

stream::ptr sycl_lz_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<sycl_lz_stream>(*this, config);
}

stream::ptr sycl_lz_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    return std::make_shared<sycl_lz_stream>(*this, config, handle);
}

std::shared_ptr<cldnn::engine> create_sycl_lz_engine(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<sycl_lz_engine>(device, runtime_type);
}

::sycl::context& sycl_lz_engine::get_sycl_context() const {
    OPENVINO_ASSERT(sycl_context != nullptr);

    return *sycl_context;
}

const ::sycl_lz::UsmHelper& sycl_lz_engine::get_usm_helper() const {
    auto sycl_lz_device = std::dynamic_pointer_cast<sycl_lz::sycl_lz_device>(_device);
    OPENVINO_ASSERT(sycl_lz_device, "[GPU] Invalid device type for sycl_lz::sycl_lz_device");
    return sycl_lz_device->get_usm_helper();
}

memory_ptr sycl_lz_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    // Todo refer: memory::ptr ocl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset)
    GPU_DEBUG_LOG << "layout=" << layout << ", type=" << type << ", reset = " << reset << std::endl;

    // auto X = sycl::malloc_shared<float>(length, q);

    try {
        memory::ptr res = nullptr;
        if (layout.format.is_image_2d()) {
            GPU_DEBUG_LOG << "Not implemented. layout.format = " << layout.format << std::endl;
            // res = std::make_shared<ocl::gpu_image2d>(this, layout);
        } else if (type == allocation_type::cl_mem) {
            GPU_DEBUG_LOG << "Not implemented. layout.format = " << layout.format << std::endl;
            // res = std::make_shared<sycl_lz::gpu_buffer>(this, layout);
            // Remote tensor need it. just use usm replace it.
            res = std::make_shared<sycl_lz::gpu_usm>(this, layout, allocation_type::usm_device);
        } else {
            res = std::make_shared<sycl_lz::gpu_usm>(this, layout, type);
        }

        if (reset || res->is_memory_reset_needed(layout)) {
            GPU_DEBUG_LOG << "Need to reset usm memory via res->fill." << std::endl;
            auto ev = res->fill(get_service_stream());
            if (ev) {
                get_service_stream().wait_for_events({ev});
            }
        }

        return res;
    } catch (::sycl::exception& e) {
        OPENVINO_THROW("[GPU] buffer allocation failed: ", e.what());
    }
}
memory_ptr sycl_lz_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    GPU_DEBUG_LOG << "Not implemented." << std::endl;
    return nullptr;
}
memory::ptr sycl_lz_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to create a subbuffer from a buffer allocated by a different engine");
    try {
        if (new_layout.format.is_image_2d()) {
            OPENVINO_NOT_IMPLEMENTED;
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            auto& new_buf = reinterpret_cast<const sycl_lz::gpu_usm&>(memory);
            auto ptr = new_buf.get_buffer().get();

            auto sub_buffer = ::sycl_lz::UsmMemory(get_usm_helper(), ptr, byte_offset);

            return std::make_shared<sycl_lz::gpu_usm>(this,
                                     new_layout,
                                     sub_buffer,
                                     memory.get_allocation_type(),
                                     memory.get_mem_tracker());
        } else {
            // auto buffer = reinterpret_cast<const sycl_lz::gpu_buffer&>(memory).get_buffer();
            // cl_buffer_region sub_buffer_region = { byte_offset, new_layout.get_linear_size() };
            // auto sub_buffer = buffer.createSubBuffer(CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR,
            //                 CL_BUFFER_CREATE_TYPE_REGION, &sub_buffer_region);

            // return std::make_shared<ocl::gpu_buffer>(this,
            //                          new_layout,
            //                          sub_buffer,
            //                          memory.get_mem_tracker());
            GPU_DEBUG_LOG << "Not implemented. gpu_buffer " << std::endl;
            return nullptr;
        }
    } catch (::sycl::exception& e) {
        OPENVINO_THROW("[GPU] create_subbuffer failed: ", e.what());
    }
}
memory_ptr sycl_lz_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to reinterpret buffer allocated by a different engine");
    OPENVINO_ASSERT(new_layout.format.is_image() == memory.get_layout().format.is_image(),
                    "[GPU] trying to reinterpret between image and non-image layouts. Current: ",
                    memory.get_layout().format.to_string(), " Target: ", new_layout.format.to_string());

    try {
        if (new_layout.format.is_image_2d()) {
            GPU_DEBUG_LOG << "Not implemented." << std::endl;
            return nullptr;
            // return std::make_shared<ocl::gpu_image2d>(this,
            //                                           new_layout,
            //                                           reinterpret_cast<const ocl::gpu_image2d&>(memory).get_buffer(),
            //                                           memory.get_mem_tracker());
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            return std::make_shared<sycl_lz::gpu_usm>(this,
                                                      new_layout,
                                                      reinterpret_cast<const sycl_lz::gpu_usm&>(memory).get_buffer(),
                                                      memory.get_allocation_type(),
                                                      memory.get_mem_tracker());
        } else {
            GPU_DEBUG_LOG << "Not implemented." << std::endl;
            return nullptr;
            // return std::make_shared<ocl::gpu_buffer>(this,
            //                                          new_layout,
            //                                          reinterpret_cast<const ocl::gpu_buffer&>(memory).get_buffer(),
            //                                          memory.get_mem_tracker());
        }
    } catch (sycl::exception const& e) {
        OPENVINO_THROW("[GPU] reinterpret_buffer failed: ", e.what());
    }

    return nullptr;
}

bool sycl_lz_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    if (!memory_capabilities::is_usm_type(mem1.get_allocation_type())) {
        GPU_DEBUG_LOG << "Not implemented. sycl_lz_engine::is_the_same_buffer" << std::endl;
        // return (reinterpret_cast<const sycl_lz::gpu_buffer&>(mem1).get_buffer() ==
        //         reinterpret_cast<const sycl_lz::gpu_buffer&>(mem2).get_buffer());
    } else {
        return (reinterpret_cast<const sycl_lz::gpu_usm&>(mem1).get_buffer() ==
                reinterpret_cast<const sycl_lz::gpu_usm&>(mem2).get_buffer());
    }

    return false;
}

bool sycl_lz_engine::check_allocatable(const layout& layout, allocation_type type) {
    OPENVINO_ASSERT(supports_allocation(type) || type == allocation_type::cl_mem,
                    "[GPU] Unsupported allocation type: ",
                    type);

    bool exceed_allocatable_mem_size = (layout.bytes_count() > get_device_info().max_alloc_mem_size);

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_allocatable_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

    GPU_DEBUG_LOG << "layout = " << layout << std::endl;

    OPENVINO_ASSERT(!exceed_allocatable_mem_size,
                    "[GPU] Exceeded max size of memory object allocation: ",
                    "requested ",
                    layout.bytes_count(),
                    " bytes, "
                    "but max alloc size supported by device is ",
                    get_device_info().max_alloc_mem_size,
                    " bytes.",
                    "Please try to reduce batch size or use lower precision.");

    auto used_mem =
        get_used_device_memory(allocation_type::usm_device) + get_used_device_memory(allocation_type::usm_host);
    auto exceed_available_mem_size = (layout.bytes_count() + used_mem > get_max_memory_size());

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_available_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

#ifdef __unix__
    // Prevent from being killed by Ooo Killer of Linux
    OPENVINO_ASSERT(!exceed_available_mem_size,
                    "[GPU] Exceeded max size of memory allocation: ",
                    "Required ",
                    layout.bytes_count(),
                    " bytes, already occupied : ",
                    used_mem,
                    " bytes, ",
                    "but available memory size is ",
                    get_max_memory_size(),
                    " bytes");
#else
    if (exceed_available_mem_size) {
        GPU_DEBUG_COUT << "[Warning] [GPU] Exceeded max size of memory allocation: " << "Required "
                       << layout.bytes_count() << " bytes, already occupied : " << used_mem
                       << " bytes, but available memory size is " << get_max_memory_size() << " bytes" << std::endl;
        GPU_DEBUG_COUT << "Please note that performance might drop due to memory swap." << std::endl;
        return false;
    }
#endif

    return true;
}

void* sycl_lz_engine::get_user_context() const {
    GPU_DEBUG_LOG << "Not implemented. sycl_lz_engine::get_user_context()" << std::endl;

    return nullptr;
}

allocation_type sycl_lz_engine::detect_usm_allocation_type(const void* memory) const {
    return use_unified_shared_memory() ? sycl_lz::gpu_usm::detect_allocation_type(this, memory)
                                       : allocation_type::unknown;
}

bool sycl_lz_engine::extension_supported(std::string extension) const {
    GPU_DEBUG_LOG << "Not implemented." << std::endl;
    return false;
}

stream& sycl_lz_engine::get_service_stream() const {
    return *_service_stream;
}

kernel::ptr sycl_lz_engine::prepare_kernel(const kernel::ptr kernel) const {
    GPU_DEBUG_LOG << "Temp implemented. sycl_lz_engine::prepare_kernel return kernel directly." << std::endl;
    // OPENVINO_ASSERT(downcast<const cldnn::sycl_lz::sycl_lz_kernel*>(kernel.get()) != nullptr);
    return kernel;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void sycl_lz_engine::create_onednn_engine(const ExecutionConfig& config)  {
    GPU_DEBUG_LOG << "Temp implemented. create_onednn_engine" << std::endl;
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_device->get_info().vendor_id == INTEL_VENDOR_ID, "[GPU] OneDNN engine can be used for Intel GPUs only");

    if (!_onednn_engine) {
        const auto& cache_dir = config.get_cache_dir();
        if (!cache_dir.empty()) {
            GPU_DEBUG_LOG << "Not implemented. oneDNN sycl_interop don't support make_engine from cache." << std::endl;
        }

        auto casted_dev = dynamic_cast<sycl_lz::sycl_lz_device*>(_device.get());
        auto device = casted_dev->get_device();
        auto context = casted_dev->get_context();
        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(device, context));
        GPU_DEBUG_LOG << "_onednn_engine = " << _onednn_engine << std::endl;
    }
}
// Returns onednn engine object which shares device and context with current engine
dnnl::engine& sycl_lz_engine::get_onednn_engine() const {
    OPENVINO_ASSERT(_onednn_engine,
                    "[GPU] Can't get onednn engine handle as it was not initialized. Please check that "
                    "create_onednn_engine() was called");
    return *_onednn_engine;
}
#endif

}  // namespace sycl_lz
}  // namespace cldnn
