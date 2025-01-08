// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_cache_sycl_lz.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/file_util.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ocl/ocl_kernel.hpp"
#include "openvino/util/pp.hpp"
#include "sycl_lz/sycl_lz_device.hpp"
#include "sycl_lz/sycl_lz_kernel.hpp"

namespace syclex = sycl::ext::oneapi::experimental;

#ifdef WIN32
#    include <sdkddkver.h>
#    ifdef NTDDI_WIN10_RS5
#        include <appmodel.h>
#    endif
#endif

#include <cassert>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#if defined(__unix__) && !defined(__ANDROID__)
#    include <malloc.h>
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include "gpu/intel/microkernels/fuser.hpp"
#endif

namespace {
std::mutex cacheAccessMutex;
}  // namespace

namespace cldnn {
// static std::mutex kernels_cache::_mutex;

std::vector<std::string> sycl_lz_kernels_cache::get_kernel_id_from_source(const std::string& sources) {
    const std::string key = "Kernel name:";
    const int key_len = key.length();
    std::vector<std::string> kernel_ids;
    size_t start_pos = 0;
    for (;;) {
        std::size_t found = sources.find(key, start_pos);
        if (found == std::string::npos) {
            break;
        }

        std::size_t start = found + key_len;
        for (; start < sources.length(); start++) {
            // std::cout << "start sources[" << start << "] = " << sources[start] << std::endl;
            if (sources[start] == ' ') {
                continue;
            }
            break;
        }

        std::size_t end = start;
        for (; end < sources.length(); end++) {
            // std::cout << "end sources[" << end << "] = " << sources[end] << std::endl;
            if (sources[end] == '\n' || sources[end] == ' ') {
                break;
            }
        }
        start_pos = end + 1;

        // static int idx = 0;
        // FILE* pf = fopen((std::to_string(idx++) + ".log").c_str(), "wb");
        // fwrite(sources.c_str(), sizeof(char), sources.length(), pf);
        // fclose(pf);
        auto kernel_id = sources.substr(start, end - start);
        kernel_ids.push_back(kernel_id);
    }

    OPENVINO_ASSERT(kernel_ids.size() > 0, "[GPU] Can't find KERNEL_ID in kernel source.");
    return kernel_ids;
}

std::vector<std::pair<std::string, kernel::ptr>> sycl_lz_kernels_cache::build_sycl_lz_kernel(
    const std::string& sources,
    const std::vector<std::string>& entry_points) {
    GPU_DEBUG_LOG << "== Build OpenCL kernel based on SYCL runtime." << std::endl;
    auto& sycl_lz_device = dynamic_cast<const sycl_lz::sycl_lz_device&>(*_device);
    auto sycl_context = sycl_lz_device.get_context();

    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclex::create_kernel_bundle_from_source(sycl_context, syclex::source_language::opencl, sources);

    // Compile and link the kernel from the source definition.
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe = syclex::build(kb_src);

    // Get a "kernel" object representing the kernel defined in the
    // source string.
    std::vector<std::pair<std::string, kernel::ptr>> kernels;
    for (auto entry_point : entry_points) {
        sycl::kernel k = kb_exe.ext_oneapi_get_kernel(entry_point);
        kernel::ptr kernel =
            std::make_shared<sycl_lz::sycl_lz_kernel>(sycl_lz::sycl_lz_kernel_type(k, sycl_lz_device.get_usm_helper()),
                                                      entry_point);
        kernels.emplace_back(std::make_pair(entry_point, kernel));
    }

    return kernels;
}

// TODO: This build_batch method should be backend specific
void sycl_lz_kernels_cache::build_batch(const batch_program& batch, compiled_kernels& compiled_kernels) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sycl_lz_kernels_cache::build_batch");

    bool dump_sources = batch.dump_custom_program;
    std::string dump_sources_dir = "";
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_sources.empty()) {
        dump_sources = true;
        dump_sources_dir = debug_config->dump_sources;
    }

    std::string err_log;  // accumulated build log from all program's parts (only contains messages from parts which

    std::string current_dump_file_name = "";
    if (dump_sources) {
        current_dump_file_name = dump_sources_dir;
        if (!current_dump_file_name.empty() && current_dump_file_name.back() != '/')
            current_dump_file_name += '/';

        current_dump_file_name += "SYCL_LZ_program_" + std::to_string(_prog_id) + "_bucket_" +
                                  std::to_string(batch.bucket_id) + "_part_" + std::to_string(batch.batch_id) + "_" +
                                  std::to_string(batch.hash_value) + ".cl";
    }

    std::ofstream dump_file;
    if (dump_sources) {
        dump_file.open(current_dump_file_name);
        if (dump_file.good()) {
            for (auto& s : batch.source)
                dump_file << s;
        }
    }

    std::string cached_bin_name = get_cache_path() + std::to_string(batch.hash_value) + ".cl_cache";
    cl::Program::Binaries precompiled_kernels = {};

    if (is_cache_enabled()) {
        // Try to load file with name ${hash_value}.cl_cache which contains precompiled kernels for current bucket
        // If read is successful, then remove kernels from compilation bucket
        GPU_DEBUG_LOG << "Not implemented. Sycl kernel can be cached via ENV currently." << std::endl;
        // std::vector<uint8_t> bin;
        // {
        //     std::lock_guard<std::mutex> lock(cacheAccessMutex);
        //     bin = ov::util::load_binary(cached_bin_name);
        // }
        // if (!bin.empty()) {
        //     precompiled_kernels.push_back(bin);
        // }
    }
    try {
        // cl::vector<cl::Kernel> kernels;
        std::string sources;

        // Run compilation
        if (precompiled_kernels.empty()) {
            GPU_DEBUG_LOG << "=============================" << std::endl;

            // cl::Program program(sycl_lz_device.get_context(), batch.source);
            {
                // OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin,
                //                    "KernelsCache::BuildProgram::RunCompilation");
                // if (program.build({sycl_lz_device.get_device()}, batch.options.c_str()) != CL_SUCCESS)
                //     throw std::runtime_error("Failed in building program.");

                for (auto s : batch.source) {
                    sources += s;
                }
            }

            // if (dump_sources && dump_file.good()) {
            //     dump_file << "\n/* Build Log:\n";
            //     for (auto& p : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>())
            //         dump_file << p.second << "\n";
            //     dump_file << "*/\n";
            // }

            if (batch.has_microkernels) {
#ifdef ENABLE_ONEDNN_FOR_GPU
                OPENVINO_ASSERT(batch.kernels_counter == 1);
                // Do we need full source code here (with batch headers)?

                GPU_DEBUG_LOG << "Not implemented." << std::endl;
                // program = fuse_microkernels(sycl_lz_device.get_context(),
                //                             sycl_lz_device.get_device(),
                //                             program,
                //                             batch.source.back());
#else   // ENABLE_ONEDNN_FOR_GPU
                OPENVINO_THROW("[GPU] Can't compile kernel w/ microkernels as onednn is not available");
#endif  // ENABLE_ONEDNN_FOR_GPU
            }

            // program.createKernels(&kernels);

            if (is_cache_enabled()) {
                // If kernels caching is enabled, then we save compiled bucket to binary file with name
                // ${code_hash_value}.cl_cache Note: Bin file contains full bucket, not separate kernels, so kernels
                // reuse across different models is quite limited Bucket size can be changed in
                // get_max_kernels_per_batch() method, but forcing it to 1 will lead to much longer compile time.
                std::lock_guard<std::mutex> lock(cacheAccessMutex);
                // ov::intel_gpu::save_binary(cached_bin_name, getProgramBinaries(program));
                GPU_DEBUG_LOG << "Not implemented. Sycl kernel can be cached via ENV currently." << std::endl;
            }
        } else {
            GPU_DEBUG_LOG << "Not implemented. Sycl doesn't cache kerenl explicitly." << std::endl;
            // cl::Program program(sycl_lz_device.get_context(), {sycl_lz_device.get_device()}, precompiled_kernels);
            // if (program.build({sycl_lz_device.get_device()}, batch.options.c_str()) != CL_SUCCESS)
            //     throw std::runtime_error("Failed in building program with a precompiled kernel.");

            // program.createKernels(&kernels);
        }

        {
            std::lock_guard<std::mutex> lock(_mutex);
            {
                auto entry_points = get_kernel_id_from_source(sources);
                auto kernels = build_sycl_lz_kernel(sources, entry_points);
                for (auto& kernel : kernels) {
                    GPU_DEBUG_LOG << "Find entry_point from sources: " << kernel.first << std::endl;
                    const auto& iter = batch.entry_point_to_id.find(kernel.first);
                    if (iter != batch.entry_point_to_id.end()) {
                        auto& params = iter->second.first;
                        auto kernel_part_idx = iter->second.second;
                        if (compiled_kernels.find(params) != compiled_kernels.end()) {
                            compiled_kernels[params].push_back(std::make_pair(kernel.second, kernel_part_idx));
                            // std::cout << "  == cache compiled kernel: " << kernel.first << std::endl;
                        } else {
                            compiled_kernels[params] = {std::make_pair(kernel.second, kernel_part_idx)};
                            // std::cout << "  == cache compiled kernel: " << kernel.first << std::endl;
                        }
                        if (_kernel_batch_hash.find(params) == _kernel_batch_hash.end()) {
                            _kernel_batch_hash[params] = batch.hash_value;
                        }
                    } else {
                        throw std::runtime_error("Could not find entry point");
                    }
                }
            }
        }
    } catch (const sycl::exception& err) {
        if (dump_sources && dump_file.good())
            dump_file << "\n/* Build Log:\n";

        if (dump_sources && dump_file.good())
            dump_file << err.what() << "\n";
        err_log += std::string(err.what()) + '\n';

        if (dump_sources && dump_file.good())
            dump_file << "*/\n";
    }
    if (!err_log.empty()) {
        GPU_DEBUG_INFO << "-------- Sycl OpenCL build error" << std::endl;
        GPU_DEBUG_INFO << err_log << std::endl;
        GPU_DEBUG_INFO << "-------- End of Sycl OpenCL build error" << std::endl;
        std::stringstream err_ss(err_log);
        std::string line;
        std::stringstream err;
        int cnt = 0;

        while (std::getline(err_ss, line, '\n')) {
            if (line.find("error") != std::string::npos)
                cnt = 5;
            cnt--;
            if (cnt > 0)
                err << line << std::endl;
            else if (cnt == 0)
                err << "...." << std::endl;
        }

        throw std::runtime_error("Program build failed(" + std::to_string(batch.bucket_id) + +"_part_" +
                                 std::to_string(batch.batch_id) + "):\n" + err.str());
    }
}

}  // namespace cldnn
