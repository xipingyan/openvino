// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_lz_stream.hpp"

#include "intel_gpu/runtime/utils.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "sycl_lz_engine.hpp"
#include "sycl_lz_event.hpp"

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
    GPU_DEBUG_LOG << "== sycl_lz_stream::sycl_lz_stream, create sycl_queue" << std::endl;
}

sycl_lz_stream::sycl_lz_stream(const sycl_lz_engine& engine, const ExecutionConfig& config, void* handle)
    : stream(config.get_property(ov::intel_gpu::queue_type), stream::get_expected_sync_method(config)),
      _engine(engine) {
    auto dev = std::dynamic_pointer_cast<sycl_lz_device>(engine.get_device());
    OPENVINO_ASSERT(dev, "Cast to sycl_lz_device fail.");

    sycl_queue = cldnn::make_unique<::sycl::queue>(dev->get_device());
    GPU_DEBUG_LOG << "== sycl_lz_stream::sycl_lz_stream, create sycl_queue" << std::endl;
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
    DEBUG_PRINT("Not implemented. sycl_lz_stream::enqueue_kernel");
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
    if (events.empty())
        return;

    for (auto& ev : events) {
        if (!ev)
            continue;
        if (auto sycl_lz_base_ev = downcast<sycl_lz_base_event>(ev.get())) {
            sycl_lz_base_ev->get().wait();
        }
    }
    DEBUG_PRINT("sycl_lz_stream::wait_for_events finish. Temp solution");
    // bool needs_barrier = false;
    // std::vector<sycl_lz_event> clevents;
    // for (auto& ev : events) {
    //     if (!ev)
    //         continue;

    //     if (auto sycl_lz_base_ev = downcast<sycl_lz_base_event>(ev.get())) {
    //         if (sycl_lz_base_ev->get().get() != nullptr) {
    //             clevents.push_back(sycl_lz_base_ev->get().get());
    //         } else {
    //             needs_barrier = true;
    //         }
    //     }
    // }

    // sycl::event barrier_ev;
    // if (needs_barrier) {
    //     try {
    //         _command_queue.enqueueBarrierWithWaitList(nullptr, &barrier_ev);
    //         clevents.push_back(barrier_ev.get());
    //     } catch (cl::Error const& err) {
    //         OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    //     }
    // }

    // if (!clevents.empty()) {
    //     auto err = clWaitForEvents(static_cast<cl_uint>(clevents.size()), &clevents[0]);
    //     if (err != CL_SUCCESS) {
    //         OPENVINO_THROW("[GPU] clWaitForEvents failed with ", err, " code");
    //     }
    // }
}
void sycl_lz_stream::enqueue_barrier() {
    DEBUG_PRINT("Not implemented. sycl_lz_stream::enqueue_barrier");
}
event::ptr sycl_lz_stream::create_user_event(bool set) {
    DEBUG_PRINT("Not implemented. sycl_lz_stream::create_user_event");
    return nullptr;
}
event::ptr sycl_lz_stream::create_base_event() {
    DEBUG_PRINT("Not implemented. sycl_lz_stream::create_base_event");
    return nullptr;
}

event::ptr sycl_lz_stream::create_base_event(sycl::event event) {
    return std::make_shared<sycl_lz::sycl_lz_event>(event, ++_queue_counter);
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
