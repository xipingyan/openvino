// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstring>
#include <string>
#include <vector>

// Control whether or not tp save tracing log.
#define ENABLE_TRACE_LOG 1

// Control whether or not to print memory usage to a log file.
#include <features.h>
#if __GLIBC_PREREQ(2, 34)
// If glibc > 2.34, it removes __malloc_hook, so we have to disable memory statistic.
#    define ENABLE_TRACE_MEM_USAGE 0
#else
#    define ENABLE_TRACE_MEM_USAGE 1
#endif

#ifndef CAT
#    define TOKEN_PASTE(x, y) x##y
#    define CAT(x, y)         TOKEN_PASTE(x, y)
#endif  // CAT

#if ENABLE_TRACE_LOG

struct MyStr {
    char _str[64] = {0};
    void set(const char* str) {
        if (str != nullptr) {
            std::memcpy(_str, str, std::min(63lu, strlen(str)));
        }
    }
    char* get() {
        return _str;
    }
    MyStr& operator=(MyStr& other) {
        if (other._str != this->_str) {
            std::memcpy(this->_str, other._str, 64);
        }
        return *this;
    }
};

class MyProfile {
public:
    MyProfile() = delete;
    MyProfile(const char* name, const char* key = nullptr, const char* val = nullptr);
    ~MyProfile();

protected:
    MyStr _name;
    uint64_t _ts1;
    MyStr _key;
    MyStr _val;
};

class MyProfileMem {
public:
    MyProfileMem() = delete;
    MyProfileMem(const char* name, const char* key = nullptr, const char* val = nullptr);
    ~MyProfileMem();

private:
    MyStr _name;
    uint64_t _ts1;
    MyStr _key;
    MyStr _val;
    int64_t _mem_sz = 0;
};

// Add line NO.
// #define ADD_LINE_NO(NAME) (NAME + std::string(":") + std::to_string(__LINE__))
#    define ADD_LINE_NO(NAME) NAME

#    define MY_PROFILE_VAR(VAR, NAME)               auto VAR = MyProfile(ADD_LINE_NO(NAME))
#    define MY_PROFILE_VAR_ARG(VAR, NAME, key, val) auto VAR = MyProfile(ADD_LINE_NO(NAME), key, val)
#    if ENABLE_TRACE_MEM_USAGE
#        define MY_PROFILE_VAR_MEM(VAR, NAME) auto VAR = MyProfileMem(ADD_LINE_NO(NAME))
#    endif

// Example 2: MY_PROFILE_VAR / MY_PROFILE_VAR_ARG
/******************************************************
MY_PROFILE_VAR(p, "fun_name")
Or
{
    MY_PROFILE_VAR(p1, "fun_name")
    func()
}
Or
{
    MY_PROFILE_VAR_ARG(p2, "fun_name", {{"arg1", "sleep 30 ms"}});
    func()
}
******************************************************/
#else
#    define MY_PROFILE_VAR(VAR, NAME)
#    define MY_PROFILE_VAR_ARG(VAR, NAME, key, val)
#    define MY_PROFILE_VAR_MEM(VAR, NAME, key, val)
#endif