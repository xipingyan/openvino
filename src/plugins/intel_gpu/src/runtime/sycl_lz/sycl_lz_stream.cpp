// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_lz_stream.hpp"

#include "intel_gpu/runtime/utils.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "sycl_lz_engine.hpp"
#include "sycl_lz_event.hpp"

// OpenCL kernel
#include "ocl/ocl_kernel.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

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

::sycl::queue& sycl_lz_stream::get_sycl_queue() const {
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
    DEBUG_PRINT("Not implemented. set_arguments no need at present.");
    // static std::mutex m;
    // std::lock_guard<std::mutex> guard(m);

    // auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);
    // GPU_DEBUG_LOG << "sycl_lz_stream::set_arguments, ocl_kernel id = " << ocl_kernel.get_id() << std::endl;
    // auto& kern = ocl_kernel.get_handle();

    // try {
    //     GPU_DEBUG_TRACE_DETAIL << "Set arguments for primitive: " << args_desc.layerID << " (" << kernel.get_id() << " = " << kern.get() << ")\n";
    //     set_arguments_impl(kern, args_desc.arguments, args);
    // } catch (cl::Error const& err) {
    //     OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    // }
}

inline sycl::range<3> toSyclRange(const std::vector<size_t>& v) {
    switch (v.size()) {
        case 1:
            return sycl::range(1, 1, v[0]);
        case 2:
            return sycl::range(1, v[0], v[1]);
        case 3:
            return sycl::range(v[0], v[1], v[2]);
        default:
            return sycl::range{1, 1, 1};
    }
}

event::ptr sycl_lz_stream::enqueue_kernel(kernel& kernel,
                                          const kernel_arguments_desc& args_desc,
                                          const kernel_arguments_data& args,
                                          std::vector<event::ptr> const& deps,
                                          bool is_output) {
    DEBUG_PRINT("Temp implemented. enqueue OCL kernel via SYCL.");
    auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);

    auto& kern = ocl_kernel.get_handle();

    sycl::nd_range ndr =
        sycl::nd_range{toSyclRange(args_desc.workGroups.global), toSyclRange(args_desc.workGroups.local)};
    GPU_DEBUG_LOG << "sycl::nd_range global_range=[" << ndr.get_global_range()[0] << ", " << ndr.get_global_range()[1]
                  << ", " << ndr.get_global_range()[2] << "], local_range=[" << ndr.get_local_range()[0] << ", "
                  << ndr.get_local_range()[1] << ", " << ndr.get_local_range()[2] << "]" << std::endl;

    // Kernel defined as an OpenCL C string.  This could be dynamically
    // generated instead of a literal.
    OPENVINO_ASSERT(ocl_kernel.get_kernel_source().size() > 0u);
    std::string source;
    for (auto s : ocl_kernel.get_kernel_source()) {
        source += s;
    }
    std::cout << "  == source = \n" << source << std::endl;

    std::cout << "  == Start to kernel_bundle opencl source" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclex::create_kernel_bundle_from_source(
            sycl_queue->get_context(),
            syclex::source_language::opencl,
            source);

    // Compile and link the kernel from the source definition.
    std::cout << "  == Start to build OpenCL kernel and kernel_bundle kb_src" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclex::build(kb_src);

    // Get a "kernel" object representing the kernel defined in the
    // source string.
    std::cout << "  == Start to get sycl::kernel, ocl_kernel.get_id() = " << ocl_kernel.get_id() << std::endl;

    sycl::kernel k = kb_exe.ext_oneapi_get_kernel(ocl_kernel.get_id());

    std::vector<sycl::event> dep_events;
    std::vector<sycl::event>* dep_events_ptr = nullptr;
    if (m_sync_method == SyncMethods::events) {
        for (auto& dep : deps) {
            if (auto ocl_base_ev = std::dynamic_pointer_cast<sycl_lz_base_event>(dep)) {
                // if (ocl_base_ev->get() != nullptr)
                dep_events.push_back(ocl_base_ev->get());
            }
        }
        dep_events_ptr = &dep_events;
    } else if (m_sync_method == SyncMethods::barriers) {
        // sync_events(deps, is_output);
        DEBUG_PRINT("Not implemented. m_sync_method == SyncMethods::barriers.");
    }

    // Unify all inputs.
    std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>> inputs_buf;
    for (size_t i = 0; i < args.inputs.size(); i++) {
        // GPU_DEBUG_LOG << "  == args_desc.arguments[i].t = " << args_desc. << std::endl;
        sycl::buffer params_buf(static_cast<uint8_t*>(args.inputs[i]->buffer_ptr()),
                                sycl::range{args.inputs[i]->size()});
        bool is_output = args_desc.arguments[i].t == argument_desc::Types::OUTPUT;
        inputs_buf.push_back({params_buf, is_output});
    }

    std::cout << "  == Start to submit" << std::endl;
    auto ret_ev = sycl_queue->submit([&](sycl::handler& cgh) {
        cgh.depends_on(dep_events);
        for (size_t i = 0; i < inputs_buf.size(); i++) {
            if (inputs_buf[i].second) {
                sycl::accessor acc_param{inputs_buf[i].first, cgh, sycl::read_write};
                cgh.set_arg(i, acc_param);
            } else {
                sycl::accessor acc_param{inputs_buf[i].first, cgh, sycl::read_only};
                cgh.set_arg(i, acc_param);
            }
        }

        // Invoke the kernel over an nd-range.
        cgh.parallel_for(ndr, k);
    });

    return std::make_shared<sycl_lz_event>(ret_ev, ++_queue_counter);
    // return nullptr;
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
    DEBUG_PRINT("sycl_lz_stream::wait_for_events finish. Temp solution, events.size=" << events.size());
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
