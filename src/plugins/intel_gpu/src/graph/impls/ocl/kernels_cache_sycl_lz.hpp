// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernels_cache.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>

#include "openvino/runtime/threading/itask_executor.hpp"


namespace cldnn {

class sycl_lz_kernels_cache : public kernels_cache {
public:
private:
    // void get_program_source(const kernels_code& kernels_source_code, std::vector<batch_program>*) const;
    void build_batch(const batch_program& batch, compiled_kernels& compiled_kernels);

    std::vector<std::string> get_kernel_id_from_source(const std::string& sources);

    std::vector<std::pair<std::string, kernel::ptr>> build_sycl_lz_kernel(const std::string& sources,
                                                                          const std::vector<std::string>& entry_point,
                                                                          const std::string& options);

public:
    explicit sycl_lz_kernels_cache(engine& engine,
                                   const ExecutionConfig& config,
                                   uint32_t prog_id,
                                   std::shared_ptr<ov::threading::ITaskExecutor> task_executor = nullptr,
                                   const std::map<std::string, std::string>& batch_headers = {})
        : kernels_cache(engine, config, prog_id, task_executor, batch_headers) {}
};

}  // namespace cldnn
