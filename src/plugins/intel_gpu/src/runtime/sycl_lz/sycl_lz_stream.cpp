// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_lz_stream.hpp"

#include "intel_gpu/runtime/utils.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "sycl_lz_engine.hpp"

namespace cldnn {
namespace sycl_lz {

sycl_lz_stream::sycl_lz_stream(const sycl_lz_engine& engine, const ExecutionConfig& config)
    : stream(config.get_property(ov::intel_gpu::queue_type), stream::get_expected_sync_method(config)),
      _engine(engine) {
    // : ocl_stream(engine, config) {
    // sycl_queue = cldnn::make_unique<::sycl::queue>(
    //     ::sycl::make_queue<::sycl::backend::opencl>(get_cl_queue().get(), engine.get_sycl_context()));
    auto dev = std::dynamic_pointer_cast<sycl_lz_device>(engine.get_device());
    OPENVINO_ASSERT(dev, "Cast to sycl_lz_device fail.");

    sycl_queue = cldnn::make_unique<::sycl::queue>(dev->get_device());
}

sycl_lz_stream::sycl_lz_stream(const sycl_lz_engine& engine, const ExecutionConfig& config, void* handle)
    : stream(config.get_property(ov::intel_gpu::queue_type), stream::get_expected_sync_method(config)),
      _engine(engine) {
    DEBUG_PRINT("Not implemented.");

    auto dev = std::dynamic_pointer_cast<sycl_lz_device>(engine.get_device());
    OPENVINO_ASSERT(dev, "Cast to sycl_lz_device fail.");

    sycl_queue = cldnn::make_unique<::sycl::queue>(dev->get_device());
}

::sycl::queue& sycl_lz_stream::get_sycl_queue() {
    OPENVINO_ASSERT(sycl_queue != nullptr);
    return *sycl_queue;
}

void sycl_lz_stream::flush() const {
    DEBUG_PRINT("Not implemented.");
}
void sycl_lz_stream::finish() const {
    DEBUG_PRINT("Not implemented.");
}
void sycl_lz_stream::wait() {
    DEBUG_PRINT("Not implemented.");
}

void sycl_lz_stream::set_arguments(kernel& kernel,
                                   const kernel_arguments_desc& args_desc,
                                   const kernel_arguments_data& args) {
    DEBUG_PRINT("Not implemented.");
}
event::ptr sycl_lz_stream::enqueue_kernel(kernel& kernel,
                                          const kernel_arguments_desc& args_desc,
                                          const kernel_arguments_data& args,
                                          std::vector<event::ptr> const& deps,
                                          bool is_output) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}
event::ptr sycl_lz_stream::enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}
event::ptr sycl_lz_stream::group_events(std::vector<event::ptr> const& deps) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}
void sycl_lz_stream::wait_for_events(const std::vector<event::ptr>& events) {
    DEBUG_PRINT("Not implemented.");
}
void sycl_lz_stream::enqueue_barrier() {
    DEBUG_PRINT("Not implemented.");
}
event::ptr sycl_lz_stream::create_user_event(bool set) {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}
event::ptr sycl_lz_stream::create_base_event() {
    DEBUG_PRINT("Not implemented.");
    return nullptr;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& sycl_lz_stream::get_onednn_stream() {
    OPENVINO_ASSERT(m_queue_type == QueueTypes::in_order,
                    "[GPU] Can't create onednn stream handle as onednn doesn't support out-of-order queue");
    OPENVINO_ASSERT(_engine.get_device_info().vendor_id == INTEL_VENDOR_ID,
                    "[GPU] Can't create onednn stream handle as for non-Intel devices");
    if (!_onednn_stream) {
        DEBUG_PRINT("Not implemented.");
        // auto r = dnnl_sycl_interop_stream_create(&_onednn_stream, _engine.get_onednn_engine(), &_command_queue);
        // OPENVINO_ASSERT(r == dnnl_success, "[GPU] dnnl_sycl_interop_stream_create fail.");
    }
    return *_onednn_stream;
}
#endif

QueueTypes sycl_lz_stream::detect_queue_type(void* queue_handle) {
    sycl::queue* q = static_cast<sycl::queue*>(queue_handle);
    return q->is_in_order() ? QueueTypes::in_order : QueueTypes::out_of_order;
}

}  // namespace sycl_lz
}  // namespace cldnn
