// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory.hpp"

#include <memory>
#include <vector>

namespace cldnn {

struct work_group_sizes {
    std::vector<size_t> global;
    std::vector<size_t> local;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct scalar_desc {
    union ValueT {
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        int8_t s8;
        int16_t s16;
        int32_t s32;
        int64_t s64;
        float f32;
        double f64;
    };

    enum class Types {
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
    };

    Types t;
    ValueT v;
};

using scalars_desc = std::vector<scalar_desc>;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ArgumentDescpirtor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct argument_desc {
    enum class Types {
        INPUT,
        OUTPUT,
        WEIGHTS,
        BIAS,
        SCALE_TABLE,
        SLOPE,
        INTERNAL_BUFFER,
        SCALAR,
        CELL,       // LSTM cell input
        WEIGHTS_ZERO_POINTS,
        ACTIVATIONS_ZERO_POINTS,
        COMPENSATION,
        INPUT_OF_FUSED_PRIMITIVE,
        SHAPE_INFO
    };

    Types t;
    uint32_t index;
};

inline std::ostream& operator<<(std::ostream& os, const argument_desc::Types& type) {
#define CASE_ITM(ITM) \
    case ITM:         \
        os << #ITM;   \
        break
    switch (type) {
        CASE_ITM(argument_desc::Types::ACTIVATIONS_ZERO_POINTS);
        CASE_ITM(argument_desc::Types::BIAS);
        CASE_ITM(argument_desc::Types::CELL);
        CASE_ITM(argument_desc::Types::COMPENSATION);
        CASE_ITM(argument_desc::Types::INPUT);
        CASE_ITM(argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE);
        CASE_ITM(argument_desc::Types::INTERNAL_BUFFER);
        CASE_ITM(argument_desc::Types::OUTPUT);
        CASE_ITM(argument_desc::Types::SCALAR);
        CASE_ITM(argument_desc::Types::SCALE_TABLE);
        CASE_ITM(argument_desc::Types::SHAPE_INFO);
        CASE_ITM(argument_desc::Types::SLOPE);
        CASE_ITM(argument_desc::Types::WEIGHTS);
        CASE_ITM(argument_desc::Types::WEIGHTS_ZERO_POINTS);
    default:
        os << "unknown";
        break;
    }

    return os;
}

using arguments_desc = std::vector<argument_desc>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct kernel_arguments_desc {
    work_group_sizes workGroups;
    arguments_desc arguments;
    scalars_desc scalars;
    std::string layerID;
};

struct kernel_arguments_data {
    std::vector<memory::cptr> inputs;
    std::vector<memory::cptr> intermediates;
    std::vector<memory::cptr> outputs;
    memory::cptr weights;
    memory::cptr recurrent;
    memory::cptr hidden;
    memory::cptr cell;
    memory::cptr bias;
    memory::cptr weights_zero_points;
    memory::cptr activations_zero_points;
    memory::cptr compensation;
    memory::cptr lookup_table;
    memory::cptr scale_table;
    memory::cptr slope;
    memory::cptr shape_info;

    std::vector<memory::cptr> fused_op_inputs;
    const scalars_desc* scalars = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelString
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct kernel_string {
    std::string str;
    std::string jit;
    std::string undefs;
    std::string options;
    std::string entry_point;
    bool batch_compilation;
    bool has_microkernels;

    kernel_string() : str(""), jit(""), undefs(""), options(""), entry_point(""), batch_compilation(false), has_microkernels(false) {}

    std::string get_str() const { return str + jit + undefs + options + entry_point; }
    size_t get_hash() const { return std::hash<std::string>()(get_str()); }
};

}  // namespace cldnn
