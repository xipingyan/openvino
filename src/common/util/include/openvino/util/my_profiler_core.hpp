// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <vector>

class MyProfileCore {
public:
    MyProfileCore() = delete;
    // MyProfileCore(MyProfileCore&)=delete;
    MyProfileCore(const std::string& name,
              const std::vector<std::pair<std::string, std::string>>& args =
                  std::vector<std::pair<std::string, std::string>>());
    ~MyProfileCore();

private:
    std::string _name;
    uint64_t _ts1;
    std::vector<std::pair<std::string, std::string>> _args;
};

#define MY_PROFILE_CORE(NAME)           MyProfileCore(NAME + std::string(":") + std::to_string(__LINE__))
#define MY_PROFILE_CORE_ARGS(NAME, ...) MyProfileCore(NAME + std::string(":") + std::to_string(__LINE__), __VA_ARGS__)

// Example:
// ==========================================
// auto p = MyProfileCore("fun_name")
// Or
// {
//     auto p = MyProfileCore("fun_name")
//     func()
// }
// Or
// {
//     auto p2 = MyProfileCore("fun_name", {{"arg1", "sleep 30 ms"}});
//     func()
// }