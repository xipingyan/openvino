// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_lz_device_detector.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "sycl_lz_device.hpp"
// #include "ocl_common.hpp"

#include "sycl/sycl.hpp"

#include <string>
#include <vector>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#    pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace {

// The priority return by this function impacts the order of devices reported by GPU plugin and devices enumeration
// Lower priority value means lower device ID
// Current behavior is: Intel iGPU < Intel dGPU < any other GPU
// Order of Intel dGPUs is undefined and depends on the OCL impl
// Order of other vendor GPUs is undefined and depends on the OCL impl
size_t get_device_priority(const cldnn::device_info& info) {
    if (info.vendor_id == cldnn::INTEL_VENDOR_ID && info.dev_type == cldnn::device_type::integrated_gpu) {
        return 0;
    } else if (info.vendor_id == cldnn::INTEL_VENDOR_ID) {
        return 1;
    } else {
        return std::numeric_limits<size_t>::max();
    }
}
}  // namespace

namespace cldnn {
namespace sycl_lz {

// static std::vector<cl::Device> getSubDevices(cl::Device& rootDevice) {
//     cl_uint maxSubDevices;
//     size_t maxSubDevicesSize;
//     const auto err = clGetDeviceInfo(rootDevice(),
//                                      CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
//                                      sizeof(maxSubDevices),
//                                      &maxSubDevices,
//                                      &maxSubDevicesSize);

//     OPENVINO_ASSERT(err == CL_SUCCESS && maxSubDevicesSize == sizeof(maxSubDevices),
//                     "[GPU] clGetDeviceInfo(..., CL_DEVICE_PARTITION_MAX_SUB_DEVICES,...)");
//     if (maxSubDevices == 0) {
//         return {};
//     }

//     const auto partitionProperties = rootDevice.getInfo<CL_DEVICE_PARTITION_PROPERTIES>();
//     const auto partitionable = std::any_of(partitionProperties.begin(),
//                                            partitionProperties.end(),
//                                            [](const decltype(partitionProperties)::value_type& prop) {
//                                                return prop == CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
//                                            });

//     if (!partitionable) {
//         return {};
//     }

//     const auto partitionAffinityDomain = rootDevice.getInfo<CL_DEVICE_PARTITION_AFFINITY_DOMAIN>();
//     const decltype(partitionAffinityDomain) expectedFlags =
//         CL_DEVICE_AFFINITY_DOMAIN_NUMA | CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;

//     if ((partitionAffinityDomain & expectedFlags) != expectedFlags) {
//         return {};
//     }

//     std::vector<cl::Device> subDevices;
//     cl_device_partition_property partitionProperty[] = {CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
//                                                         CL_DEVICE_AFFINITY_DOMAIN_NUMA,
//                                                         0};

//     rootDevice.createSubDevices(partitionProperty, &subDevices);

//     return subDevices;
// }

std::vector<device::ptr> sycl_lz_device_detector::sort_devices(const std::vector<device::ptr>& devices_list) {
    std::vector<device::ptr> sorted_list = devices_list;
    std::stable_sort(sorted_list.begin(), sorted_list.end(), [](device::ptr d1, device::ptr d2) {
        return get_device_priority(d1->get_info()) < get_device_priority(d2->get_info());
    });

    return sorted_list;
}

std::map<std::string, device::ptr> sycl_lz_device_detector::get_available_devices(void* user_context,
                                                                                  void* user_device,
                                                                                  int ctx_device_id,
                                                                                  int target_tile_id) const {
    std::vector<device::ptr> devices_list;
    if (user_context != nullptr) {
        devices_list = create_device_list_from_user_context(user_context, ctx_device_id);
    } else if (user_device != nullptr) {
        devices_list = create_device_list_from_user_device(user_device);
    } else {
        devices_list = create_device_list();
    }

    devices_list = sort_devices(devices_list);

    std::map<std::string, device::ptr> ret;
    uint32_t idx = 0;
    for (auto& dptr : devices_list) {
        auto map_id = std::to_string(idx++);
        ret[map_id] = dptr;

        // auto root_device = std::dynamic_pointer_cast<ocl_device>(dptr);
        // OPENVINO_ASSERT(root_device != nullptr, "[GPU] Invalid device type created in ocl_device_detector");

        // auto sub_devices = getSubDevices(root_device->get_device());
        // if (!sub_devices.empty()) {
        //     uint32_t sub_idx = 0;
        //     for (auto& sub_device : sub_devices) {
        //         if (target_tile_id != -1 && static_cast<int>(sub_idx) != target_tile_id) {
        //             sub_idx++;
        //             continue;
        //         }
        //         auto sub_device_ptr =
        //             std::make_shared<ocl_device>(sub_device, cl::Context(sub_device), root_device->get_platform());
        //         ret[map_id + "." + std::to_string(sub_idx++)] = sub_device_ptr;
        //     }
        // }
    }
    return ret;
}

std::vector<device::ptr> sycl_lz_device_detector::create_device_list() const {
    std::vector<device::ptr> supported_devices;
    for (auto platform : sycl::platform::get_platforms()) {
        try {
            static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
            // oneAPI 2024
            static constexpr auto INTEL_PLATFORM_NAME_2024 = "Intel(R) Level-Zero";
            // oneAPI 2025
            static constexpr auto INTEL_PLATFORM_NAME_2025 = "Intel(R) oneAPI Unified Runtime over Level-Zero";

            std::vector<sycl_lz::sycl_lz_device> devices;
            if (platform.get_info<sycl::info::platform::vendor>() != INTEL_PLATFORM_VENDOR) {
                continue;
            }

            if (!one_of(platform.get_info<sycl::info::platform::name>(), {INTEL_PLATFORM_NAME_2024, INTEL_PLATFORM_NAME_2025})) {
                continue;
            }

            for (auto device : platform.get_devices()) {
                sycl::queue q(device);
                if (q.get_backend() != sycl::backend::ext_oneapi_level_zero) {
                    continue;
                }

                supported_devices.emplace_back(std::make_shared<sycl_lz_device>(device, q.get_context(), platform));
            }
        } catch (std::exception& ex) {
            GPU_DEBUG_LOG << "Devices query/creation failed for " << platform.get_info<sycl::info::platform::name>()
                          << ": " << ex.what() << std::endl;
            GPU_DEBUG_LOG << "Platform is skipped" << std::endl;
            continue;
        }
    }

    GPU_DEBUG_TRACE << "supported_devices num: " << supported_devices.size() << std::endl;
    return supported_devices;
}

std::vector<device::ptr> sycl_lz_device_detector::create_device_list_from_user_context(void* user_context,
                                                                                       int ctx_device_id) const {
    std::vector<device::ptr> supported_devices;
    OPENVINO_ASSERT(!supported_devices.empty(), "Not implemented[SYCL_RUNTIME]. [GPU] User defined context does not have supported GPU device.");
    return supported_devices;
}

std::vector<device::ptr> sycl_lz_device_detector::create_device_list_from_user_device(void* user_device) const {
    std::vector<device::ptr> supported_devices;
    OPENVINO_ASSERT(!supported_devices.empty(), "Not implemented[SYCL_RUNTIME]. [GPU] User specified device is not supported.");
    return std::vector<device::ptr>();
}

}  // namespace sycl_lz
}  // namespace cldnn
