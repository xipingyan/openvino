// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dump_profile.hpp"

#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "hook_mem_record.hpp"

#ifdef _WIN32
#    include <Windows.h>
#    include <intrin.h>
#else
#    include <dlfcn.h>
#    include <x86intrin.h>
#endif
#pragma intrinsic(__rdtsc)

#if ENABLE_TRACE_MEM_USAGE
static int64_t g_min_mem_size = 0x7FFFFFFFFFFFFFFF;
struct dump_mem_items {
    std::string name;  // The name of the event, as displayed in Trace Viewer
    // std::string ph = "C";  // The event type, 'B'/'E' OR 'X'
    // std::string pid = "0"; // The process ID for the process
    std::string tid;
    uint64_t ts = 0;
    // Args
    std::string mem_key;
    int64_t mem_sz = 0;
};
#endif  // ENABLE_TRACE_MEM_USAGE

struct dump_items {
    MyStr name;  // The name of the event, as displayed in Trace Viewer
    // MyStr cat = "PERF";       // The event categories
    // MyStr ph = "X";  // The event type, 'B'/'E' OR 'X'
    // MyStr pid = "0"; // The process ID for the process
    std::thread::id tid = std::this_thread::get_id();  // The thread ID for the thread that output this event.
    uint64_t ts1 = 0;                                  // The tracing clock timestamp of the event, [microsecond]
    uint64_t ts2 = 0;                                  // Duration = ts2 - ts1.
    // std::string tts;       // Optional. The thread clock timestamp of the event
    MyStr key;  // Avoiding to malloc/free, just keep a pair.
    MyStr val;
};

static inline std::string get_thread_id() {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}

static uint64_t rdtsc_calibrate(int seconds = 1) {
    uint64_t start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return (__rdtsc() - start_ticks) / seconds;
}

inline std::string location(void) {
#ifdef _WIN32
    TCHAR path[1024];
    ::GetModuleFileNameA(NULL, path, 1024);
    std::string dll_fn = std::string(path);
    std::string fn = dll_fn.substr(dll_fn.find_last_of("\\") + 1, dll_fn.length());
#else
    Dl_info info;
    dladdr(reinterpret_cast<void*>(&location), &info);
    std::string dll_fn = std::string(info.dli_fname);
    std::string fn = dll_fn.substr(dll_fn.find_last_of("/") + 1, dll_fn.length());
#endif

    fn = fn.substr(0, fn.find_last_of("."));
    return fn;
}

class ProfilerManager {
protected:
    std::vector<dump_items> _vecItems;
#if ENABLE_TRACE_MEM_USAGE
    std::vector<dump_mem_items> _vecMemItems;
#endif
    std::atomic<uint64_t> tsc_ticks_per_second{0};
    std::atomic<uint64_t> tsc_ticks_base{0};
    std::mutex _mutex;

public:
    ProfilerManager() {
        _vecMemItems.reserve(1024 * 1024);
        _vecItems.reserve(1024 * 1024);

        if (tsc_ticks_per_second == 0) {
            uint64_t expected = 0;
            auto tps = rdtsc_calibrate();
            tsc_ticks_per_second.compare_exchange_strong(expected, tps);
            std::cout << "=== ProfilerManager: tsc_ticks_per_second = " << tsc_ticks_per_second << std::endl;
            tsc_ticks_base.compare_exchange_strong(expected, __rdtsc());
            std::cout << "=== ProfilerManager: tsc_ticks_base = " << tsc_ticks_base << std::endl;
        }
    }

    int64_t get_mem_size() {
        return _vecItems.capacity() * sizeof(dump_items) + _vecMemItems.capacity() * sizeof(dump_mem_items);
    }

    ProfilerManager(ProfilerManager& other) = delete;
    void operator=(const ProfilerManager&) = delete;
    ~ProfilerManager() {
        // Save tracing log to json file.
        save_to_json();
    }

    void add(const dump_items& val) {
        std::lock_guard<std::mutex> lk(_mutex);
        _vecItems.emplace_back(val);
    }

#if ENABLE_TRACE_MEM_USAGE
    void add_mem(const dump_mem_items& val) {
        std::lock_guard<std::mutex> lk(_mutex);
        _vecMemItems.emplace_back(val);
    }
#endif  // ENABLE_TRACE_MEM_USAGE

private:
    std::string tsc_to_nsec(uint64_t tsc_ticks) {
        double val = (tsc_ticks - tsc_ticks_base) * 1000000.0 / tsc_ticks_per_second;
        return std::to_string(val);
    }

    std::string tsc_to_nsec(uint64_t start, uint64_t end) {
        double val = (end - start) * 1000000.0 / tsc_ticks_per_second;
        return std::to_string(val);
    }

    void save_to_json() {
        std::string json_fn = "profile_" + location() + ".json";
        FILE* pf;
#ifdef _WIN32
        errno_t err = fopen_s(&pf, json_fn.c_str(), "wb");
        if (err != 0) {
            printf("Can't fopen:%s", json_fn.c_str());
            return;
        }
#else
        pf = fopen(json_fn.c_str(), "wb");
        if (nullptr == pf) {
            printf("Can't fopen:%s", json_fn.c_str());
            return;
        }
#endif
        // Headers
        fprintf(pf, "{\n\"schemaVersion\": 1,\n\"traceEvents\":[\n");

// Save mem satistic.
#if ENABLE_TRACE_MEM_USAGE
        for (size_t i = 0; i < _vecMemItems.size(); i++) {
            auto& itm = _vecMemItems[i];
            // Write 1 event
            fprintf(pf, "{");
            fprintf(pf, "\"name\":\"%s\",", itm.name.c_str());
            fprintf(pf, "\"ph\":\"C\",");
            fprintf(pf, "\"pid\":\"0\",");
            fprintf(pf, "\"ts\":\"%s\",", tsc_to_nsec(itm.ts).c_str());
            fprintf(pf, "\"args\":{");
            if (itm.mem_sz) {
                fprintf(pf, "\"%s\":\"%ld\" ", itm.mem_key.c_str(), itm.mem_sz - g_min_mem_size);
            }
            fprintf(pf, "}},\n");
        }
        fprintf(pf, "{}%s\n", (_vecItems.size() == 0) ? "" : ",");
#endif

        // Save tracing log:
        // MyStr cat;       // The event categories
        // MyStr ph = "X";  // The event type, 'B'/'E' OR 'X'
        // MyStr pid = "0"; // The process ID for the process

        for (size_t i = 0; i < _vecItems.size(); i++) {
            auto& itm = _vecItems[i];
            // Write 1 event
            fprintf(pf, "{");
            fprintf(pf, "\"name\":\"%s\",", itm.name._str);
            fprintf(pf, "\"cat\":\"PERF\",");
            fprintf(pf, "\"ph\":\"X\",");
            fprintf(pf, "\"pid\":\"0\",");
            fprintf(pf, "\"tid\":\"%u\",", *static_cast<unsigned int*>(static_cast<void*>(&itm.tid)));
            fprintf(pf, "\"ts\":\"%s\",", tsc_to_nsec(itm.ts1).c_str());
            fprintf(pf, "\"dur\":\"%s\",", tsc_to_nsec(itm.ts1, itm.ts2).c_str());
            fprintf(pf, "\"args\":{");
            fprintf(pf, "\"%s\":\"%s\"", itm.key._str, itm.val._str);
            fprintf(pf, "}},\n");
        }
        if (_vecItems.size() > 0) {
            fprintf(pf, "{}\n");
        }

        fprintf(pf, "]\n}\n");
        fclose(pf);
        printf("Profiler log is saved to: %s\n", json_fn.c_str());
    }
};

static ProfilerManager g_profileManage;

#if ENABLE_TRACE_LOG
MyProfile::MyProfile(const char* name, const char* key, const char* val) {
    _name.set(name);
    _key.set(key);
    _val.set(val);
    _ts1 = __rdtsc();
}

MyProfile::~MyProfile() {
    dump_items itm;
    itm.ts2 = __rdtsc();
    itm.ts1 = _ts1;
    itm.name = _name;
    itm.tid = std::this_thread::get_id();
    itm.key = _key;
    itm.val = _val;
    g_profileManage.add(itm);
}
#endif  // !ENABLE_TRACE_LOG

#if ENABLE_TRACE_MEM_USAGE
inline int64_t get_allocated_mem_size() {
    auto tmp = (int64_t)RecordMem::getInstance()->getMemSize() - g_profileManage.get_mem_size();
    g_min_mem_size = std::min(g_min_mem_size, tmp);
    return tmp;
}

MyProfileMem::MyProfileMem(const char* name, const char* key, const char* val) {
    _name.set(name);
    _key.set(key);
    _val.set(val);
    _ts1 = __rdtsc();
    _mem_sz = get_allocated_mem_size();
}

MyProfileMem::~MyProfileMem() {
    int64_t mem_diff = get_allocated_mem_size() - _mem_sz;

    // Start
    dump_mem_items itm1;
    itm1.ts = _ts1;
    itm1.name = "mem";
    itm1.tid = get_thread_id();
    itm1.mem_key = "cats";
    itm1.mem_sz = _mem_sz;
    g_profileManage.add_mem(itm1);
    // End
    dump_mem_items itm2;
    itm2.ts = __rdtsc();
    itm2.name = "mem";
    itm2.tid = get_thread_id();
    itm2.mem_key = "cats";
    itm2.mem_sz = get_allocated_mem_size();
    g_profileManage.add_mem(itm2);

    // Tracing log
    dump_items itm;
    itm.ts2 = __rdtsc();
    itm.ts1 = _ts1;
    itm.name = _name;
    itm.key = _key;
    itm.val = _val;
    g_profileManage.add(itm);

    // Diff
    // dump_mem_items itm2;
    // itm2.ts = _ts1;
    // itm2.name = "mem_diff";
    // itm2.tid = get_thread_id();
    // itm2.mem_key="cats";
    // itm2.mem_sz=mem_diff;
    // g_profileManage.add_mem(itm2);
}
#endif  // !ENABLE_TRACE_MEM_USAGE