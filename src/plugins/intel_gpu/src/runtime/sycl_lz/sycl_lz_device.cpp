// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
#ifndef NOMINMAX
# define NOMINMAX
#endif
#include "gpu/intel/jit/jit_generator.hpp"
#include "gpu/intel/jit/ngen/ngen_level_zero.hpp"
#endif  // ENABLE_ONEDNN_FOR_GPU

#include "sycl_lz_common.hpp"
#include "sycl_lz_device.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <cassert>
#include <time.h>
#include <limits>
#include <chrono>
#include <fstream>
#include <iostream>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <setupapi.h>
#include <devguid.h>
#include <cstring>
#else
#include <unistd.h>
#include <limits.h>
#include <link.h>
#include <dlfcn.h>
#endif


namespace cldnn {
namespace sycl_lz {

namespace {

#ifdef ENABLE_ONEDNN_FOR_GPU
gpu_arch convert_ngen_arch(ngen::HW gpu_arch) {
    switch (gpu_arch) {
        case ngen::HW::Gen9: return gpu_arch::gen9;
        case ngen::HW::Gen11: return gpu_arch::gen11;
        case ngen::HW::XeLP: return gpu_arch::xe_lp;
        case ngen::HW::XeHP: return gpu_arch::xe_hp;
        case ngen::HW::XeHPG: return gpu_arch::xe_hpg;
        case ngen::HW::XeHPC: return gpu_arch::xe_hpc;
        case ngen::HW::Xe2: return gpu_arch::xe2;
        case ngen::HW::Xe3: return gpu_arch::xe3;
        case ngen::HW::Gen10:
        case ngen::HW::Unknown: return gpu_arch::unknown;
    }
    return gpu_arch::unknown;
}
#endif

int driver_dev_id() {
    const std::vector<int> unused_ids = {
        0x4905, 0x4906, 0x4907, 0x4908
    };
    std::vector<int> result;

#ifdef _WIN32
    {
        HDEVINFO device_info_set = SetupDiGetClassDevsA(&GUID_DEVCLASS_DISPLAY, NULL, NULL, DIGCF_PRESENT);
        if (device_info_set == INVALID_HANDLE_VALUE)
            return 0;

        SP_DEVINFO_DATA devinfo_data;
        std::memset(&devinfo_data, 0, sizeof(devinfo_data));
        devinfo_data.cbSize = sizeof(devinfo_data);

        for (DWORD dev_idx = 0; SetupDiEnumDeviceInfo(device_info_set, dev_idx, &devinfo_data); dev_idx++) {
            const size_t kBufSize = 512;
            char buf[kBufSize];
            if (!SetupDiGetDeviceInstanceIdA(device_info_set, &devinfo_data, buf, kBufSize, NULL)) {
                continue;
            }

            char* vendor_pos = std::strstr(buf, "VEN_");
            if (vendor_pos != NULL && std::stoi(vendor_pos + 4, NULL, 16) == 0x8086) {
                char* device_pos = strstr(vendor_pos, "DEV_");
                if (device_pos != NULL) {
                    result.push_back(std::stoi(device_pos + 4, NULL, 16));
                }
            }
        }

        if (device_info_set) {
            SetupDiDestroyDeviceInfoList(device_info_set);
        }
    }
#elif defined(__linux__)
    {
        std::string dev_base{ "/sys/devices/pci0000:00/0000:00:02.0/" };
        std::ifstream ifs(dev_base + "vendor");
        if (ifs.good()) {
            int ven_id;
            ifs >> std::hex >> ven_id;
            ifs.close();
            if (ven_id == 0x8086) {
                ifs.open(dev_base + "device");
                if (ifs.good()) {
                    int res = 0;
                    ifs >> std::hex >> res;
                    result.push_back(res);
                }
            }
        }
    }
#endif

    auto id_itr = result.begin();
    while (id_itr != result.end()) {
        if (std::find(unused_ids.begin(), unused_ids.end(), *id_itr) != unused_ids.end())
            id_itr = result.erase(id_itr);
        else
            id_itr++;
    }

    if (result.empty())
        return 0;
    else
        return result.back();
}

device_type get_device_type(const sycl::device& device) {
    device.get_info<sycl::info::device::driver_version>();
    auto unified_mem = device.get_info<sycl::info::device::host_unified_memory>();

    return unified_mem ? device_type::integrated_gpu : device_type::discrete_gpu;
}

// gfx_version parse_version(cl_uint gmdid) {
//     union GMDID {
//         uint32_t value;
//         struct {
//             uint32_t revision : 6;
//             uint32_t reserved : 8;
//             uint32_t release : 8;
//             uint32_t architecture : 10;
//         };
//     };

//     GMDID gmd_id = {gmdid};
//     if (gmd_id.architecture > 0 && gmd_id.architecture < 100) {
//         // New format
//         return { static_cast<uint16_t>(gmd_id.architecture), static_cast<uint8_t>(gmd_id.release), static_cast<uint8_t>(gmd_id.revision)};
//     } else {
//         // Old format
//         cl_uint ver = gmdid;
//         uint16_t major = ver >> 16;
//         uint8_t minor = (ver >> 8) & 0xFF;
//         uint8_t revision = ver & 0xFF;

//         return {major, minor, revision};
//     }
// }

bool get_imad_support(const sycl::device& device) {
    std::string dev_name = device.get_info<sycl::info::device::name>();

    if (dev_name.find("Gen12") != std::string::npos ||
        dev_name.find("Xe") != std::string::npos)
        return true;

    if (get_device_type(device) == device_type::integrated_gpu) {
        const std::vector<int> imad_ids = {
            0x9A40, 0x9A49, 0x9A59, 0x9AD9,
            0x9A60, 0x9A68, 0x9A70, 0x9A78,
            0x9A7F, 0x9AF8, 0x9AC0, 0x9AC9
        };
        int dev_id = driver_dev_id();
        if (dev_id == 0)
            return false;

        if (std::find(imad_ids.begin(), imad_ids.end(), dev_id) != imad_ids.end())
            return true;
    } else {
        return true;
    }

    return false;
}

device_info init_device_info(const sycl::device& device) {
    device_info info = {};
    DEBUG_PRINT("init_device_info");

    info.vendor_id = static_cast<uint32_t>(device.get_info<sycl::info::device::vendor_id>());
    info.dev_name = device.get_info<sycl::info::device::name>();
    info.driver_version = device.get_info<sycl::info::device::driver_version>();
    info.dev_type = get_device_type(device);

    GPU_DEBUG_INFO << "GPU type: " << (info.dev_type == device_type::discrete_gpu ? "discrete_gpu" : "integrated_gpu")
                   << ". info.dev_name = " << info.dev_name << ", info.driver_version = " << info.driver_version
                   << std::endl;

    info.execution_units_count = device.get_info<sycl::info::device::max_compute_units>();

    info.gpu_frequency = static_cast<uint32_t>(device.get_info<sycl::info::device::max_clock_frequency>());

    info.max_work_group_size = static_cast<uint64_t>(device.get_info<sycl::info::device::max_work_group_size>());

    // For some reason nvidia runtime throws an exception (CL_INVALID_KERNEL_ARGS) for WG as follows:
    // global: < 1 x 32 x 5184 >
    // local: < 1 x 1 x 576 >
    // While local  < 1 x 1 x 36 > works fine
    // So below we limit max WG size by 64 which was selected based on few experiments.
    constexpr int nvidia_vendor_id = 0x10DE;
    if (info.vendor_id == nvidia_vendor_id) {
        info.max_work_group_size = 64;
    }

    info.max_local_mem_size = static_cast<uint64_t>(device.get_info<sycl::info::device::local_mem_size>());
    info.max_global_mem_size = static_cast<uint64_t>(device.get_info<sycl::info::device::global_mem_size>());
    info.max_alloc_mem_size = static_cast<uint64_t>(device.get_info<sycl::info::device::max_mem_alloc_size>());

    info.supports_image = static_cast<uint8_t>(device.get_info<sycl::info::device::image_support>());
    info.max_image2d_width = static_cast<uint64_t>(device.get_info<sycl::info::device::image2d_max_width>());
    info.max_image2d_height = static_cast<uint64_t>(device.get_info<sycl::info::device::image2d_max_height>());

    // Check for supported features.
    std::vector<std::string> extensions = device.get_info<sycl::info::device::extensions>();
    auto in_extensions = [&](std::string ext) {
        return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
    };

    info.supports_intel_planar_yuv = in_extensions("cl_intel_planar_yuv");
    info.supports_fp16 = in_extensions("cl_khr_fp16");
    info.supports_fp64 = in_extensions("cl_khr_fp64");

    auto half_fp_config = device.get_info<sycl::info::device::half_fp_config>();
    auto found_denorm =
        std::find(half_fp_config.begin(), half_fp_config.end(), sycl::info::fp_config::denorm) != half_fp_config.end();
    info.supports_fp16_denorms = info.supports_fp16 && found_denorm;

    info.supports_khr_subgroups = in_extensions("cl_khr_subgroups");
    info.supports_intel_subgroups = in_extensions("cl_intel_subgroups");
    info.supports_intel_subgroups_short = in_extensions("cl_intel_subgroups_short");
    info.supports_intel_subgroups_char = in_extensions("cl_intel_subgroups_char");
    info.supports_intel_required_subgroup_size = in_extensions("cl_intel_required_subgroup_size");

    info.supports_imad = get_imad_support(device);
    info.supports_immad = false;

    info.supports_usm = in_extensions("cl_intel_unified_shared_memory") ||
                        in_extensions("cl_intel_unified_shared_memory_preview");

    info.supports_local_block_io = in_extensions("cl_intel_subgroup_local_block_io");

    info.supports_queue_families = in_extensions("cl_intel_command_queue_families");

    if (info.supports_intel_required_subgroup_size) {
        // SUB_GROUP_SIZES_INTEL
        info.supported_simd_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    } else {
        // Set these values as reasonable default for most of the supported platforms
        info.supported_simd_sizes = {8, 16, 32};
    }

    bool device_uuid_supported = in_extensions("cl_khr_device_uuid ") != std::string::npos;
    if (device_uuid_supported) {
        static_assert(CL_UUID_SIZE_KHR == ov::device::UUID::MAX_UUID_SIZE, "");
        static_assert(CL_LUID_SIZE_KHR == ov::device::LUID::MAX_LUID_SIZE, "");
        std::fill_n(std::begin(info.luid.luid),
                    ov::device::LUID::MAX_LUID_SIZE,
                    0);  // Not implemented[SYCL_RUNTIME]. SYCL doesn't expose LUID.
        info.uuid.uuid = device.get_info<sycl::info::device::ext_intel_device_info_uuid>();
    } else {
        std::fill_n(std::begin(info.luid.luid), ov::device::LUID::MAX_LUID_SIZE, 0);
        std::fill_n(std::begin(info.uuid.uuid), ov::device::UUID::MAX_UUID_SIZE, 0);
    }

    bool device_attr_supported = in_extensions("cl_intel_device_attribute_query");
    bool nv_device_attr_supported = in_extensions("cl_nv_device_attribute_query");
    info.has_separate_cache = false;
    // if (device_attr_supported) {
    //     info.ip_version = device.get_info<sycl::info::device::IP_VERSION_INTEL>();
    //     info.gfx_ver = parse_version(info.ip_version);
    //     info.device_id = device.get_info<sycl::info::device::ID_INTEL>();
    info.num_slices = device.get_info<sycl::info::device::ext_intel_gpu_slices>();
    info.num_sub_slices_per_slice = device.get_info<sycl::info::device::ext_intel_gpu_subslices_per_slice>();
    info.num_eus_per_sub_slice = device.get_info<sycl::info::device::ext_intel_gpu_eu_count_per_subslice>();
    info.num_threads_per_eu = device.get_info<sycl::info::device::ext_intel_gpu_hw_threads_per_eu>();
    // auto features = device.get_info<sycl::info::device::FEATURE_CAPABILITIES_INTEL>();

    // Temp solution. oneDNN kernel need info.supports_immad == true.
    GPU_DEBUG_LOG << "Temp solution, hard code info.supports_immad == true, because oneDNN need it." << std::endl;
    info.supports_immad = true;

    //     info.supports_imad = info.supports_imad || (features & CL_DEVICE_FEATURE_FLAG_DP4A_INTEL);
    //     info.supports_immad = info.supports_immad || (features & CL_DEVICE_FEATURE_FLAG_DPAS_INTEL);
    // if (info.dev_type == device_type::discrete_gpu || info.gfx_ver.major > 12 ||
    //     (info.gfx_ver.major == 12 && info.gfx_ver.minor >= 70)) {
    //     info.has_separate_cache = true;
    // }
    //     GPU_DEBUG_INFO << "GPU version: " << static_cast<int>(info.gfx_ver.major) << "."
    //                    << static_cast<int>(info.gfx_ver.minor) << "." << static_cast<int>(info.gfx_ver.revision)
    //                    << (info.has_separate_cache ? " with separate cache" : "") << std::endl;
    //     GPU_DEBUG_GET_INSTANCE(debug_config);
    //     GPU_DEBUG_IF(debug_config->disable_onednn)
    //     info.supports_immad = false;
    // } else if (nv_device_attr_supported) {
    //     info.gfx_ver = {static_cast<uint16_t>(device.get_info<sycl::info::device::COMPUTE_CAPABILITY_MAJOR_NV>()),
    //                     static_cast<uint8_t>(device.get_info<sycl::info::device::COMPUTE_CAPABILITY_MINOR_NV>()),
    //                     0};
    // } else {
    // Not implemented[SYCL_RUNTIME].
    info.gfx_ver = {0, 0, 0};
    info.device_id = driver_dev_id();
    info.num_slices = 0;
    info.num_sub_slices_per_slice = 0;
    info.num_eus_per_sub_slice = 0;
    info.num_threads_per_eu = 0;
    // }

    info.num_ccs = 1;
    // Not implemented[SYCL_RUNTIME].
    // if (info.supports_queue_families) {
    //     cl_uint num_queues = 0;

    //     std::vector<cl_queue_family_properties_intel> qfprops =
    //         device.get_info<sycl::info::device::QUEUE_FAMILY_PROPERTIES_INTEL>();
    //     for (cl_uint q = 0; q < qfprops.size(); q++) {
    //         if (qfprops[q].capabilities == CL_QUEUE_DEFAULT_CAPABILITIES_INTEL && qfprops[q].count > num_queues) {
    //             num_queues = qfprops[q].count;
    //         }
    //     }
    //     info.num_ccs = std::max<uint32_t>(num_queues, info.num_ccs);
    // }

#ifdef ENABLE_ONEDNN_FOR_GPU
    using namespace dnnl::impl::gpu::intel::jit;
    ngen::HW hw = ngen::HW::Unknown;
    ngen::Product product = {ngen::ProductFamily::Unknown, 0};

    // "Not implemented[SYCL_RUNTIME]. oneDNN"
    // Because the next 3 lines codes will depend on level-zero. so temp comment it.
    auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
    auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl::queue(device).get_context());
    ngen::LevelZeroCodeGenerator<ngen::HW::Unknown>::detectHWInfo(zeContext, zeDevice, hw, product);
    // jit_generator<ngen::HW::Unknown>::detectHWInfo(context.get(), device.get(), hw, product);

    info.arch = convert_ngen_arch(hw);
    // We change the value of this flag to avoid OneDNN usage for the platforms unknown to OneDNN
    // This is required to guarantee some level of forward compatibility for the new HW generations
    // as OneDNN code generators are not generic and typically requires some updates for the new architectures
    // Ideally, we shouldn't do that as OCL impls sometimes also check this flag, but in order to avoid that
    // we need to ensure that graph transformations are not relying on this flag as indicator that onednn will be
    // used
    if (product.family == ngen::ProductFamily::Unknown) {
        info.supports_immad = false;
    }
#else   // ENABLE_ONEDNN_FOR_GPU
    info.arch = gpu_arch::unknown;
#endif  // ENABLE_ONEDNN_FOR_GPU

    return info;
}

// bool does_device_support(int32_t param, const sycl::device& device) {
    // cl_device_unified_shared_memory_capabilities_intel capabilities;
    // auto err = clGetDeviceInfo(device.get(), param, sizeof(cl_device_unified_shared_memory_capabilities_intel), &capabilities, NULL);
    // if (err) throw std::runtime_error("[CLDNN ERROR]. clGetDeviceInfo error " + std::to_string(err));

    // return !((capabilities & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL) == 0u);
// }

memory_capabilities init_memory_caps(const sycl::device& device, const device_info& info) {
    std::vector<allocation_type> memory_caps;

    // "Not implemented[SYCL_RUNTIME]. Temp init"
    memory_caps.push_back(allocation_type::usm_host);
    memory_caps.push_back(allocation_type::usm_shared);
    memory_caps.push_back(allocation_type::usm_device);

    // if (info.supports_usm) {
    //     if (does_device_support(CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL, device)) {
    //         memory_caps.push_back(allocation_type::usm_host);
    //     }
    //     if (does_device_support(CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, device)) {
    //         memory_caps.push_back(allocation_type::usm_shared);
    //     }
    //     if (does_device_support(CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, device)) {
    //         memory_caps.push_back(allocation_type::usm_device);
    //     }
    // }

    return memory_capabilities(memory_caps);
}

}  // namespace


// sycl_lz_device::sycl_lz_device()
// : _context(ctx)
// , _device(dev)
// , _platform(platform)
// , _info(init_device_info(dev, ctx))
// , _mem_caps(init_memory_caps(dev, _info))
// , _usm_helper(new cl::UsmHelper(_context, _device, use_unified_shared_memory())) {
// }

sycl_lz_device::sycl_lz_device(const sycl::device dev, const sycl::context ctx, const sycl::platform& platform)
    : _device(dev),
      _context(ctx),
      _platform(platform),
      _info(init_device_info(dev)),
      _mem_caps(init_memory_caps(dev, _info)),
      _usm_helper(new ::sycl_lz::UsmHelper(ctx, dev, use_unified_shared_memory())) {}

bool sycl_lz_device::is_same(const device::ptr other) {
    auto casted = downcast<sycl_lz_device>(other.get());
    if (!casted)
        return false;

    return _device == casted->get_device() && _platform == casted->get_platform();
}

}  // namespace sycl_lz
}  // namespace cldnn
