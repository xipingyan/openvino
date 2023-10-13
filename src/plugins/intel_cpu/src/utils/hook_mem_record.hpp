// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <iostream>
#include <mutex>
#include <thread>


#include <iostream>
#include <mutex>
#include <thread>

/*************************************************
 * RecordMem will record each malloc's size.
 * The size will be reduced when free be called.
 **************************************************/
class RecordMem {
    struct MemPair {
        void* key;
        size_t value;
    };
    static const size_t COUNTER_SIZE = 1024 * 1024 * 1024;
    static RecordMem* instancePtr;
    MemPair* memPairs = nullptr;
    size_t idx1 = 0;
    size_t idx2 = 0;

    std::mutex _mutex;
    size_t current_mem_size = 0;
    size_t mem_peak = 0;

    RecordMem() {
        // Do not print.
        memPairs = new MemPair[COUNTER_SIZE];
        if (memPairs == nullptr) {
            printf("Error: can't malloc buf size %lu\n", COUNTER_SIZE);
            exit(0);
        }
    }

    void addMemSize(const size_t& sz) {
        std::lock_guard<std::mutex> lk(_mutex);
        current_mem_size += sz;
        mem_peak = std::max(current_mem_size, mem_peak);
#ifdef _DEBUG
        printf("-->Add: cur_sz=%lu, add_sz=%lu, peak_sz=%lu\n", current_mem_size, sz, mem_peak);
#endif
    }
    void reduceMemSize(const size_t& sz) {
        std::lock_guard<std::mutex> lk(_mutex);
        current_mem_size -= sz;
#ifdef _DEBUG
        printf("-->Reduce: cur_sz=%lu, reduce_sz=%lu, peak_sz=%lu\n", current_mem_size, sz, mem_peak);
#endif
    }

public:
    RecordMem(RecordMem&) = delete;
    static RecordMem* getInstance() {
        if (instancePtr == NULL) {
            instancePtr = new RecordMem();
            return instancePtr;
        } else {
            return instancePtr;
        }
    }

    void add(void* key, const size_t& value) {
        for (size_t i = idx1; i < COUNTER_SIZE; i++) {
            if (memPairs[i].key == 0) {
                memPairs[i].key = key;
                memPairs[i].value = value;
                idx1 = std::min(i + 1, size_t(COUNTER_SIZE - 1));
                idx2 = std::max(i + 1, idx2);
                // Move idx1
                for (size_t j = i + 1; j <= idx2; j++) {
                    if (memPairs[j].key == 0) {
                        idx1 = j;
                        break;
                    }
                }
                addMemSize(value);
                return;
            }
        }
        printf("Error: There is no space to put value. Exit.\n");
        exit(0);
    }

    void remove(void* key) {
        for (size_t i = 0; i < idx2; i++) {
            if (memPairs[i].key == key) {
                idx1 = std::min(idx1, i);
                memPairs[i].key = 0;
                // Move idx2, if the key is last.
                if (i + 1 == idx2) {
                    for (size_t j = i; j >= idx1; j--) {
                        if (memPairs[j].key == 0) {
                            idx2 = j;
                        } else {
                            break;
                        }
                    }
                }
                reduceMemSize(memPairs[i].value);
                return;
            }
        }
        printf("Error: can't find key:%p\n", key);
    }

    size_t getMemSize() {
        std::lock_guard<std::mutex> lk(_mutex);
        return current_mem_size;
    }
    size_t getMemPeak() {
        std::lock_guard<std::mutex> lk(_mutex);
        return mem_peak;
    }
    void showStatus() {
        printf("-->[idx1, idx2]=[%ld, %ld]\n", idx1, idx2);
        for (size_t i = 0; i < idx2; i++) {
            printf("key, value = %p, %lu\n", memPairs[i].key, memPairs[i].value);
        }
    }
};

// Unit test.
// int main()
// {
//     CALL_FUN(_FUN(RecordgetIem::get()->semove((void*)0xFFF2))-//
//     CALL_FUN(RecordMem::getInstance()->add((void*)0xFFF4, 1)); CALL_F N(RecordMem::gL_I s ALce()->add(_void*F0xFFF5,
//     1));   CALL_FUN(RecordMem::getInstance()->remove((void*)0xFFF4));
//     CALremove((vFid*)0xFFF6));
//     reecrn 0;Mem::getInstance()->remove((void*)0xFFF6));
//     return 0;
// }   CALL_FUN(RecordMem::getInstance()->remove((void*)0xFFF3));
//     CALL_FUN(RecordMem::getInstance()->remove((void*)0xFFF6));
//     return 0;
// }