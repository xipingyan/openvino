// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/predicate.hpp"

#include "openvino/core/log_util.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

namespace ov::pass::pattern {

PatternSymbolValue::PatternSymbolValue() : m_value() {};
PatternSymbolValue::PatternSymbolValue(const std::shared_ptr<ov::Symbol>& s) : m_value(s) {};
PatternSymbolValue::PatternSymbolValue(int64_t i) : m_value(i) {};
PatternSymbolValue::PatternSymbolValue(double d) : m_value(d) {};
PatternSymbolValue::PatternSymbolValue(const std::vector<PatternSymbolValue>& g) : m_value(g) {};

bool PatternSymbolValue::is_dynamic() const {
    return is_valid() && m_value.is<std::shared_ptr<ov::Symbol>>();
}

bool PatternSymbolValue::is_static() const {
    return !is_dynamic() && !is_group();
}

bool PatternSymbolValue::is_group() const {
    return is_valid() && m_value.is<std::vector<PatternSymbolValue>>();
}

bool PatternSymbolValue::is_integer() const {
    return is_valid() && m_value.is<int64_t>();
}

bool PatternSymbolValue::is_double() const {
    return is_valid() && m_value.is<double>();
}

int64_t PatternSymbolValue::i() const {
    return m_value.as<int64_t>();
}

double PatternSymbolValue::d() const {
    return m_value.as<double>();
}

std::shared_ptr<ov::Symbol> PatternSymbolValue::s() const {
    return m_value.as<std::shared_ptr<ov::Symbol>>();
}

const std::vector<PatternSymbolValue>& PatternSymbolValue::g() const {
    return m_value.as<std::vector<PatternSymbolValue>>();
}

bool PatternSymbolValue::is_valid() const {
    if (m_value == nullptr)
        return false;
    size_t value_identities = m_value.is<int64_t>() + m_value.is<double>() + m_value.is<std::shared_ptr<ov::Symbol>>() +
                              m_value.is<std::vector<PatternSymbolValue>>();
    return value_identities == 1;
}

bool PatternSymbolValue::operator==(const ov::pass::pattern::PatternSymbolValue& other) const {
    if (is_integer() && other.is_double()) {
        return std::fabs(static_cast<double>(i()) - other.d()) < 1e-5;
    } else if (is_double() && other.is_integer()) {
        return std::fabs(d() - static_cast<double>(other.i())) < 1e-5;
    } else if (is_double() && other.is_double()) {
        return std::fabs(d() - other.d()) < 1e-5;
    } else if (is_dynamic() && other.is_dynamic()) {
        return ov::symbol::are_equal(s(), other.s());
    } else {
        return m_value == other.m_value;
    }
}

bool PatternSymbolValue::operator!=(const ov::pass::pattern::PatternSymbolValue& other) const {
    return !(*this == other);
}

namespace op {

namespace {
constexpr bool symbol_true_predicate(pass::pattern::PatternSymbolMap&, const Output<Node>&) {
    return true;
}
}  // namespace

Predicate::Predicate() : m_name("always_true"), m_pred(symbol_true_predicate) {}
Predicate::Predicate(std::nullptr_t) : Predicate() {}

bool Predicate::operator()(Matcher* m, const Output<Node>& output) const {
    bool result = m_pred(m->get_symbols(), output);
    OPENVINO_LOG_PREDICATE1(m, m_name, result);
    return result;
}

bool Predicate::operator()(const std::shared_ptr<Node>& node) const {
    OPENVINO_ASSERT(!m_requires_map, "Predicate " + m_name + " called with unexpected argument: std::shared_ptr<Node>");
    PatternSymbolMap dummy_map;
    return m_pred(dummy_map, node);
}

bool Predicate::operator()(const Output<Node>& output) const {
    OPENVINO_ASSERT(!m_requires_map, "Predicate " + m_name + " called with unexpected argument: Output<Node>");
    PatternSymbolMap dummy_map;
    return m_pred(dummy_map, output);
}
}  // namespace op
}  // namespace ov::pass::pattern
