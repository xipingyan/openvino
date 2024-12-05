// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_lz_engine.hpp"
#include "sycl_lz_stream.hpp"


namespace cldnn {
namespace sycl_lz {


sycl_lz_engine::sycl_lz_engine(const device::ptr dev, runtime_types runtime_type)
: engine(dev) {
    GPU_DEBUG_INFO << "create sycl_lz_engine" << std::endl;
    // : ocl_engine(dev, runtime_type) {
    auto casted_dev = dynamic_cast<sycl_lz::sycl_lz_device*>(_device.get());
    auto device = casted_dev->get_device();

    // _queue = cldnn::make_unique<sycl::queue>(device);
    sycl_context = cldnn::make_unique<::sycl::context>(device);

    // sycl_context = cldnn::make_unique<::sycl::context>(sycl::make_context<::sycl::backend::opencl>(get_cl_context().get()));
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

memory_ptr sycl_lz_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    // Todo refer: memory::ptr ocl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset)

    memory::ptr res = nullptr;
    // auto X = sycl::malloc_shared<float>(length, q);

    // try {
    //     memory::ptr res = nullptr;
    //     if (layout.format.is_image_2d()) {
    //         res = std::make_shared<ocl::gpu_image2d>(this, layout);
    //     } else if (type == allocation_type::cl_mem) {
    //         res = std::make_shared<ocl::gpu_buffer>(this, layout);
    //     } else {
    //         res = std::make_shared<ocl::gpu_usm>(this, layout, type);
    //     }

    //     if (reset || res->is_memory_reset_needed(layout)) {
    //         auto ev = res->fill(get_service_stream());
    //         if (ev) {
    //             get_service_stream().wait_for_events({ev});
    //         }
    //     }

    //     return res;
    // } catch (const cl::Error& clErr) {
    //     switch (clErr.err()) {
    //     case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    //     case CL_OUT_OF_RESOURCES:
    //     case CL_OUT_OF_HOST_MEMORY:
    //     case CL_INVALID_BUFFER_SIZE:
    //         OPENVINO_THROW("[GPU] out of GPU resources");
    //     default:
    //         OPENVINO_THROW("[GPU] buffer allocation failed");
    //     }
    // }
    DEBUG_PRINT("Not implemented. return null memory::ptr.");
    return res;
}
memory_ptr sycl_lz_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}
memory_ptr sycl_lz_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}
bool sycl_lz_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    DEBUG_PRINT("Not implemented.");
    return false;
}
bool sycl_lz_engine::check_allocatable(const layout& layout, allocation_type type) {
    // Todo Refer: bool ocl_engine::check_allocatable(const layout& layout, allocation_type type) {
    DEBUG_PRINT("Not implemented.");
    return false;
}

void* sycl_lz_engine::get_user_context() const {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}

allocation_type sycl_lz_engine::detect_usm_allocation_type(const void* memory) const {
    DEBUG_PRINT("Not implemented.");
    return allocation_type::usm_shared;
}

bool sycl_lz_engine::extension_supported(std::string extension) const {
    DEBUG_PRINT("Not implemented.");
    return false;
}

stream& sycl_lz_engine::get_service_stream() const {
    return *_service_stream;
}

kernel::ptr sycl_lz_engine::prepare_kernel(const kernel::ptr kernel) const {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void sycl_lz_engine::create_onednn_engine(const ExecutionConfig& config)  {
    DEBUG_PRINT("Not implemented.");
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
