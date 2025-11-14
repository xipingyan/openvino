// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/simple_math.hpp"
#include "intel_gpu/primitives/stub_custom_primitive.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_gpu {

template <typename T>
static inline std::string vecToString(std::vector<T> vec) {
    if (vec.empty())
        return "";

    std::string res = std::to_string(vec[0]);
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + std::to_string(vec[i]);
    }
    return res;
}

template <>
inline std::string vecToString<std::string>(std::vector<std::string> vec) {
    if (vec.empty())
        return "";

    std::string res = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
        res += "," + vec[i];
    }
    return res;
}

class CustomLayerAttributeVisitor : public ov::AttributeVisitor {
public:
    CustomLayerAttributeVisitor() : m_values({}) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        OPENVINO_THROW("Attribute ", name, " can't be processed\n");
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        m_values[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_values[name] = std::to_string(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<double>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint8_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint16_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint32_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_values[name] = vecToString(adapter.get());
    }

    std::map<std::string, std::string> get_parameters() const {
        return m_values;
    }

protected:
    std::map<std::string, std::string> m_values;
};

void CreateStubCustomOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    CustomLayerAttributeVisitor visitor;
    op->visit_attributes(visitor);
    auto params = visitor.get_parameters();

    auto stub_custom_primitive = cldnn::stub_custom_primitive(layerName, inputs, params);
    p.add_primitive(*op, stub_custom_primitive);
}

}  // namespace ov::intel_gpu