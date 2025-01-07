// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "sycl_lz_common.hpp"
#include "sycl_lz_ext.hpp"

#include <sycl/sycl.hpp>

namespace cldnn {
namespace sycl_lz {

struct sycl_lz_device : public device {
public:
    sycl_lz_device(const sycl::device dev, const sycl::context ctx, const sycl::platform& platform);

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    const sycl::device & get_device() const { return _device; }
    sycl::device& get_device() { return _device; }
    const sycl::platform& get_platform() const { return _platform; }
    const ::sycl_lz::UsmHelper& get_usm_helper() const { return *_usm_helper; }

    const sycl::context& get_context() const {
        return _context;
    }
    sycl::context& get_context() {
        return _context;
    }

    bool is_same(const device::ptr other) override;

    ~sycl_lz_device() = default;

private:
    sycl::device _device;
    sycl::context _context;
    sycl::platform _platform;
    device_info _info;
    memory_capabilities _mem_caps;
    std::unique_ptr<::sycl_lz::UsmHelper> _usm_helper;
};

}  // namespace sycl_lz
}  // namespace cldnn
