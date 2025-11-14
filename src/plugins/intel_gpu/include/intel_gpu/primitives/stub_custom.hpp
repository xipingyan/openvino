// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include <vector>

namespace cldnn {

/// @brief This primitive executes a custom kernel provided by the application
/// @details The application is required to provide all relevant details for executing the custom kernel
/// such as: sources, entry point, work sizes and parameter bindings.
struct stub_custom_primitive : public primitive_base<stub_custom_primitive> {
    CLDNN_DECLARE_PRIMITIVE(stub_custom_primitive)

    stub_custom_primitive() : primitive_base("", {}) {}

    /// @brief Constructs stub_custom_primitive primitive
    /// @param id This primitive id.
    /// @param input Input primitive ids.
    stub_custom_primitive(const primitive_id& id, const std::vector<input_info>& inputs, const std::map<std::string, std::string>& params)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          m_params(params) {}

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const stub_custom_primitive>(rhs);

        return m_params == rhs_casted.m_params;

        return true;
    }
    std::map<std::string, std::string> m_params;
};

}  // namespace cldnn
