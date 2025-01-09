// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "print_model_statistics.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include <memory>

namespace ov {
namespace intel_gpu {
namespace {
size_t collect_stats(const std::shared_ptr<ov::Model>& m, std::map<DiscreteTypeInfo, size_t>& ops_stat) {
    const std::vector<std::shared_ptr<ov::Node>> ops = m->get_ops();
    size_t total = ops.size();
    for (auto& op : ops) {
        const auto& tinfo = op->get_type_info();
        if (ops_stat.find(tinfo) == ops_stat.end()) {
            ops_stat[tinfo] = 0;
        }

        ops_stat[tinfo]++;

        if (auto subgraph_op = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(op)) {
            for (const auto& subgraph : subgraph_op->get_functions()) {
                total += collect_stats(subgraph, ops_stat);
            }
        }
    }

    return total;
}

}  // namespace

void PrintModelStatistics::print_model(const std::shared_ptr<ov::Model>& m) {
    std::string model_prefix = transformation_callback(nullptr) ? "gpu_" : "src_";

    std::stringstream ss;
    ss << "=======================================================\n"
       << "== Print model, name = " << m->get_friendly_name() << std::endl;

    auto print_one_op = [&ss, &model_prefix](const std::shared_ptr<ov::Node>& op, const std::string& prefix = "  ") {
#define PRINT_OP(cur_op)                                                                            \
    model_prefix << cur_op->get_friendly_name() << "[" << cur_op->get_output_element_type(0) << "]" \
                 << cur_op->get_output_partial_shape(0).to_string()

        ss << prefix << PRINT_OP(op) << " = " << op->get_type_name() << "(";
        for (size_t i = 0; i < op->get_input_size(); i++) {
            ss << PRINT_OP(op->get_input_node_shared_ptr(i)) << (i == op->get_input_size() - 1 ? ")\n" : ", ");
        }
        if (op->get_input_size() == 0) {
            ss << ")\n";
        }
    };

    const std::vector<std::shared_ptr<ov::Node>> ops = m->get_ops();
    if (ops.size() == 1) {
        print_one_op(ops[0]);
    } else if (ops.size() > 1) {
        print_one_op(ops[0]);
        for (size_t i = ops.size() - 1; i > 0; i--) {
            print_one_op(ops[i]);

            if (auto subgraph_op = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(ops[i])) {
                for (const auto& subgraph : subgraph_op->get_functions()) {
                    const std::vector<std::shared_ptr<ov::Node>> sub_ops = subgraph->get_ops();
                    for (auto& sub_op : sub_ops) {
                        print_one_op(sub_op, "    ");
                    }
                }
            }
        }
    }
    GPU_DEBUG_INFO << ss.str() << std::endl;
}

bool PrintModelStatistics::run_on_model(const std::shared_ptr<ov::Model>& m) {
    print_model(m);
    return false;

    std::map<DiscreteTypeInfo, size_t> ops_stat;
    size_t total = collect_stats(m, ops_stat);

    std::stringstream ss;
    ss << "Operations statistics:\n";
    for (auto& kv : ops_stat) {
        ss << "\t" << kv.first.version_id << "::" << kv.first.name << " " << kv.second << std::endl;
    }

    ss << "\tTotal: " << total;

    GPU_DEBUG_INFO << ss.str() << std::endl;;

    return false;
}

}  // namespace intel_gpu
}  // namespace ov
