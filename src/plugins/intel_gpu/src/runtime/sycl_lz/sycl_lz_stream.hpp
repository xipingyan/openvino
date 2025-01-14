// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl/sycl.hpp"
#include "sycl_lz_engine.hpp"

namespace cldnn {
namespace sycl_lz {

class sycl_lz_stream : public stream {
public:
    sycl_lz_stream(const sycl_lz_engine& engine, const ExecutionConfig& config);
    sycl_lz_stream(const sycl_lz_engine& engine, const ExecutionConfig& config, void* handle);

    sycl_lz_stream(sycl_lz_stream&& other)
        : stream(other.m_queue_type, other.m_sync_method),
          _engine(other._engine),
          sycl_queue(other.sycl_queue),
          _queue_counter(other._queue_counter.load()),
          _last_barrier(other._last_barrier.load()),
          _last_barrier_ev(other._last_barrier_ev) {}

    ~sycl_lz_stream() = default;

    void flush() const override;
    void finish() const override;
    void wait() override;

    void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) override;
    event::ptr enqueue_kernel(kernel& kernel,
                              const kernel_arguments_desc& args_desc,
                              const kernel_arguments_data& args,
                              std::vector<event::ptr> const& deps,
                              bool is_output = false) override;
    event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) override;
    event::ptr group_events(std::vector<event::ptr> const& deps) override;
    void wait_for_events(const std::vector<event::ptr>& events) override;
    void enqueue_barrier() override;
    event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;
    event::ptr create_base_event(sycl::event event);

    const ::sycl_lz::UsmHelper& get_usm_helper() const { return _engine.get_usm_helper(); }

    static QueueTypes detect_queue_type(void* queue_handle);

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override;
#endif

    sycl::queue& get_sycl_queue() const;

private:
    void sync_events(std::vector<event::ptr> const& deps, bool is_output = false);

    const sycl_lz_engine& _engine;
    std::atomic<uint64_t> _queue_counter{0};
    std::atomic<uint64_t> _last_barrier{0};
    sycl::event _last_barrier_ev;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _onednn_stream = nullptr;
#endif

    std::shared_ptr<::sycl::queue> sycl_queue = nullptr;
};

}  // namespace sycl_lz
}  // namespace cldnn
