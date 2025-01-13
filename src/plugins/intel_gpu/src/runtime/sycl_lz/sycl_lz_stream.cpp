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
#include "sycl_lz/sycl_lz_kernel.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

namespace cldnn {
namespace sycl_lz {

sycl_lz_stream::sycl_lz_stream(const sycl_lz_engine& engine, const ExecutionConfig& config)
    : stream(config.get_property(ov::intel_gpu::queue_type), stream::get_expected_sync_method(config)),
      _engine(engine) {
    // : ocl_stream(engine, config) {
    auto dev = std::dynamic_pointer_cast<sycl_lz_device>(engine.get_device());
    OPENVINO_ASSERT(dev, "Cast to sycl_lz_device fail.");

    sycl_queue = cldnn::make_unique<::sycl::queue>(dev->get_context(), dev->get_device());
    GPU_DEBUG_LOG << "== sycl_lz_stream::sycl_lz_stream, create sycl_queue" << std::endl;
}

sycl_lz_stream::sycl_lz_stream(const sycl_lz_engine& engine, const ExecutionConfig& config, void* handle)
    : stream(config.get_property(ov::intel_gpu::queue_type), stream::get_expected_sync_method(config)),
      _engine(engine) {
    auto dev = std::dynamic_pointer_cast<sycl_lz_device>(engine.get_device());
    OPENVINO_ASSERT(dev, "Cast to sycl_lz_device fail.");

    sycl_queue = cldnn::make_unique<::sycl::queue>(dev->get_context(), dev->get_device());
    GPU_DEBUG_LOG << "== sycl_lz_stream::sycl_lz_stream, create sycl_queue" << std::endl;
}

::sycl::queue& sycl_lz_stream::get_sycl_queue() const {
    OPENVINO_ASSERT(sycl_queue != nullptr);
    return *sycl_queue;
}

void sycl_lz_stream::flush() const {
    GPU_DEBUG_LOG << "Not implemented. sycl_lz_stream::flush()" << std::endl;
}
void sycl_lz_stream::finish() const {
    GPU_DEBUG_LOG << "Temp implemented. sycl_queue->wait();" << std::endl;
    sycl_queue->wait();
}
void sycl_lz_stream::wait() {
    GPU_DEBUG_LOG << "Not implemented. sycl_lz_stream::wait()" << std::endl;
}

cl_int set_kernel_arg_sycl_kernel(
    const std::string& kernel_id,
    std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>>& inputs_buf,
    uint32_t idx,
    cldnn::memory::cptr mem,
    bool is_output = false) {
    if (!mem)
        return CL_INVALID_ARG_VALUE;

    if (mem->get_layout().format.is_image_2d()) {
        auto buf = std::dynamic_pointer_cast<const ocl::gpu_image2d>(mem)->get_buffer();
        // auto length = std::dynamic_pointer_cast<const ocl::gpu_image2d>(mem)->size();
        GPU_DEBUG_TRACE_DETAIL << "Not implemented. kernel_id: " << kernel_id << " set arg (image) " << idx
                               << " mem: " << buf.get() << " size: " << mem->size() << std::endl;

        // sycl::buffer params_buf(static_cast<uint8_t*>(buf), sycl::range{length});
        // inputs_buf.push_back({params_buf, is_output});
        return CL_SUCCESS;
    } else if (memory_capabilities::is_usm_type(mem->get_allocation_type())) {
        sycl::buffer params_buf(static_cast<uint8_t*>(mem->buffer_ptr()), sycl::range{mem->size()});
        inputs_buf.push_back({params_buf, is_output});
        return CL_SUCCESS;
    } else {
        auto buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(mem)->get_buffer();
        // auto length = std::dynamic_pointer_cast<const ocl::gpu_buffer>(mem)->size();
        GPU_DEBUG_TRACE_DETAIL << "Not implemented. kernel_id: " << kernel_id << " set arg (buffer) " << idx << " mem: " << buf.get()
                               << " size: " << mem->size() << std::endl;
        // // return kernel.setArg(idx, buf);
        // sycl::buffer params_buf(static_cast<uint8_t*>(buf.get()), sycl::range{length});
        // inputs_buf.push_back({params_buf, is_output});
        return CL_SUCCESS;
    }

    return CL_INVALID_ARG_VALUE;
}

std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>> set_arguments_impl_sycl_kernel(
    const arguments_desc& args,
    const kernel_arguments_data& data,
    const std::string& kernel_id) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;

    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    std::vector<std::pair<sycl::buffer<uint8_t, 1, sycl::image_allocator, void>, bool>> inputs_buf;

    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        cl_int status = CL_INVALID_ARG_VALUE;
        GPU_DEBUG_LOG << "i = " << i << ", args[i].t=" << args[i].t << std::endl;
        switch (args[i].t) {
        case args_t::INPUT:
            if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.inputs[args[i].index]);
            }
            break;
        case args_t::INPUT_OF_FUSED_PRIMITIVE:
            if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.fused_op_inputs[args[i].index]);
            }
            break;
        case args_t::INTERNAL_BUFFER:
            if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.intermediates[args[i].index]);
            }
            break;
        case args_t::OUTPUT:
            if (args[i].index < data.outputs.size() && data.outputs[args[i].index]) {
                status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.outputs[args[i].index], true);
            }
            break;
        case args_t::WEIGHTS:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.weights);
            break;
        case args_t::BIAS:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.bias);
            break;
        case args_t::WEIGHTS_ZERO_POINTS:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.weights_zero_points);
            break;
        case args_t::ACTIVATIONS_ZERO_POINTS:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.activations_zero_points);
            break;
        case args_t::COMPENSATION:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.compensation);
            break;
        case args_t::SCALE_TABLE:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.scale_table);
            break;
        case args_t::SLOPE:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.slope);
            break;
        case args_t::SCALAR:
            GPU_DEBUG_LOG << " == Not implemented. args_t::SCALAR" << kernel_id << std::endl;
            // if (data.scalars && args[i].index < data.scalars->size()) {
            //     const auto& scalar = (*data.scalars)[args[i].index];
            //     switch (scalar.t) {
            //     case scalar_t::UINT8:
            //         // status = kernel.setArg(i, scalar.v.u8);
            //         sycl::buffer params_buf(static_cast<uint8_t*>(&scalar.v.u8), sycl::range{1});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (u8): " << scalar.v.u8 << "\n";
            //         break;
            //     case scalar_t::UINT16:
            //         // status = kernel.setArg(i, scalar.v.u16);
            //         sycl::buffer params_buf(static_cast<uint8_t*>(&scalar.v.u16), sycl::range{2});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (u16): " << scalar.v.u16 << "\n";
            //         break;
            //     case scalar_t::UINT32:
            //         // status = kernel.setArg(i, scalar.v.u32);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.u32), sycl::range{4});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (u32): " << scalar.v.u32 << "\n";
            //         break;
            //     case scalar_t::UINT64:
            //         // status = kernel.setArg(i, scalar.v.u64);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.u64), sycl::range{8});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (u64): " << scalar.v.u64 << "\n";
            //         break;
            //     case scalar_t::INT8:
            //         // status = kernel.setArg(i, scalar.v.s8);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.s8), sycl::range{1});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (s8): " << scalar.v.s8 << "\n";
            //         break;
            //     case scalar_t::INT16:
            //         // status = kernel.setArg(i, scalar.v.s16);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.s16), sycl::range{2});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (s16): " << scalar.v.s16 << "\n";
            //         break;
            //     case scalar_t::INT32:
            //         // status = kernel.setArg(i, scalar.v.s32);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.s32), sycl::range{4});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (s32): " << scalar.v.s32 << "\n";
            //         break;
            //     case scalar_t::INT64:
            //         // status = kernel.setArg(i, scalar.v.s64);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.s64), sycl::range{8});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (s64): " << scalar.v.s64 << "\n";
            //         break;
            //     case scalar_t::FLOAT32:
            //         // status = kernel.setArg(i, scalar.v.f32);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.f32), sycl::range{4});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (f32): " << scalar.v.f32 << "\n";
            //         break;
            //     case scalar_t::FLOAT64:
            //         // status = kernel.setArg(i, scalar.v.f64);
            //         sycl::buffer params_buf(reinterpret_cast<uint8_t*>(&scalar.v.f64), sycl::range{8});
            //         inputs_buf.push_back({params_buf, false});
            //         GPU_DEBUG_TRACE_DETAIL << "kernel_id:" << kernel_id << " set scalar " << i
            //                                << " (f64): " << scalar.v.f64 << "\n";
            //         break;
            //     default:
            //         break;
            //     }
            // }
            break;
        case args_t::CELL:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.cell);
            break;
        case args_t::SHAPE_INFO:
            status = set_kernel_arg_sycl_kernel(kernel_id, inputs_buf, i, data.shape_info);
            break;
        default:
            break;
        }

        if (status != CL_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", kernel_id: " + kernel_id +
                                     ", error code: " + std::to_string(status) + "\n");
        }
    }
    return inputs_buf;
}

void sycl_lz_stream::set_arguments(kernel& kernel,
                                   const kernel_arguments_desc& args_desc,
                                   const kernel_arguments_data& args) {
    GPU_DEBUG_LOG << "Temp implemented. set_arguments is not needed at present." << std::endl;
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
            return sycl::range(v[0], 1, 1);
        case 2:
            return sycl::range(v[1], v[0], 1);
        case 3:
            return sycl::range(v[2], v[1], v[0]);
        default:
            return sycl::range{1, 1, 1};
    }
}

event::ptr sycl_lz_stream::enqueue_kernel(kernel& kernel,
                                          const kernel_arguments_desc& args_desc,
                                          const kernel_arguments_data& args,
                                          std::vector<event::ptr> const& deps,
                                          bool is_output) {
    GPU_DEBUG_LOG << "Temp implemented. enqueue OCL kernel into SYCL runtime pipeline." << std::endl;
    auto& sycl_kernel = downcast<sycl_lz::sycl_lz_kernel>(kernel);

    auto& kern = sycl_kernel.get_handle();

    sycl::nd_range ndr =
        sycl::nd_range{toSyclRange(args_desc.workGroups.global), toSyclRange(args_desc.workGroups.local)};

    sycl::kernel k = kern.get_kernel();
    GPU_DEBUG_LOG << "kernel_id = " << sycl_kernel.get_id() << ", sycl::nd_range global_range=["
                  << ndr.get_global_range()[0] << ", " << ndr.get_global_range()[1] << ", " << ndr.get_global_range()[2]
                  << "], local_range=[" << ndr.get_local_range()[0] << ", " << ndr.get_local_range()[1] << ", "
                  << ndr.get_local_range()[2] << "]" << std::endl;

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
        GPU_DEBUG_LOG << "Not implemented. m_sync_method == SyncMethods::barriers." << std::endl;
    }

    // Unify all inputs.
    auto inputs_buf = set_arguments_impl_sycl_kernel(args_desc.arguments, args, sycl_kernel.get_id());

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
}
event::ptr sycl_lz_stream::enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) {
    GPU_DEBUG_LOG << "Not implemented. sycl_lz_stream::enqueue_marker." << std::endl;
    return nullptr;
}
event::ptr sycl_lz_stream::group_events(std::vector<event::ptr> const& deps) {
    GPU_DEBUG_LOG << "Not implemented." << std::endl;
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
    GPU_DEBUG_LOG << "sycl_lz_stream::wait_for_events finish. Temp solution, events.size=" << events.size()
                  << std::endl;
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
    GPU_DEBUG_LOG << "Not implemented. sycl_lz_stream::enqueue_barrier" << std::endl;
}
event::ptr sycl_lz_stream::create_user_event(bool set) {
    GPU_DEBUG_LOG << "Not implemented. sycl_lz_stream::create_user_event" << std::endl;
    return nullptr;
}
event::ptr sycl_lz_stream::create_base_event() {
    GPU_DEBUG_LOG << "Not implemented. sycl_lz_stream::create_base_event" << std::endl;
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
        GPU_DEBUG_LOG << "Not implemented." << std::endl;
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
