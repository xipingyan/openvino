// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "intel_gpu/runtime/optionals.hpp"
#include "sycl_lz_base_event.hpp"
#include "sycl_lz_common.hpp"

namespace cldnn {
namespace sycl_lz {

struct sycl_lz_event : public sycl_lz_base_event {
public:
    sycl_lz_event(sycl::event const& ev, uint64_t queue_stamp = 0) : sycl_lz_base_event(queue_stamp), _event(ev) {}

    sycl_lz_event(uint64_t duration_nsec, uint64_t queue_stamp = 0)
        : sycl_lz_base_event(queue_stamp),
          duration_nsec(duration_nsec) {}

    sycl::event& get() override {
        return _event;
    }

private:
    bool _callback_set = false;
    void set_sycl_lz_callback();
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    bool add_event_handler_impl(event_handler, void*) override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    static void CL_CALLBACK sycl_lz_event_completion_callback(sycl::event, cl_int, void* me);

    friend struct ocl_events;

protected:
    sycl::event _event;
    optional_value<uint64_t> duration_nsec;
};

struct sycl_lz_events : public sycl_lz_base_event {
public:
    sycl_lz_events(std::vector<event::ptr> const& ev) : sycl_lz_base_event(0) {
        process_events(ev);
    }

    sycl::event& get() override {
        return _last_sycl_lz_event;
    }

    void reset() override {
        event::reset();
        _events.clear();
    }

private:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;

    void process_events(const std::vector<event::ptr>& ev) {
        for (size_t i = 0; i < ev.size(); i++) {
            auto multiple_events = dynamic_cast<sycl_lz_events*>(ev[i].get());
            if (multiple_events) {
                for (size_t j = 0; j < multiple_events->_events.size(); j++) {
                    if (auto base_ev = dynamic_cast<sycl_lz_event*>(multiple_events->_events[j].get())) {
                        auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                        if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                            _queue_stamp = current_ev_queue_stamp;
                            _last_sycl_lz_event = base_ev->get();
                        }
                    }
                    _events.push_back(multiple_events->_events[j]);
                }
            } else {
                if (auto base_ev = dynamic_cast<sycl_lz_event*>(ev[i].get())) {
                    auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                    if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                        _queue_stamp = current_ev_queue_stamp;
                        _last_sycl_lz_event = base_ev->get();
                    }
                }
                _events.push_back(ev[i]);
            }
        }
    }

    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    sycl::event _last_sycl_lz_event;
    std::vector<event::ptr> _events;
};

}  // namespace sycl_lz
}  // namespace cldnn
