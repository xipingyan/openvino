// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <malloc.h>
#include <stdio.h>

#include <mutex>
#include <thread>

#include "dump_profile.hpp"
// Refer: https://linux.die.net/man/3/__malloc_hook

#include "hook_mem_record.hpp"
// initializing instancePtr with NULL
RecordMem* RecordMem::instancePtr = NULL;

#if ENABLE_TRACE_MEM_USAGE

// Control if print log.
#    define PRINT_LOG 0

static std::mutex g_mutex;

/* Prototypes for our hooks. */
static void my_init_hook(void);
static void* my_malloc_hook(size_t, const void*);
static void* my_realloc_hook(void*, size_t, const void*);
static void* my_memalign_hook(size_t, size_t, const void*);
static void my_free_hook(void*, const void*);
// static void my_malloc_initialize_hook(void);
// static void my_after_morecore_hook(void);

/* Variables to save original hooks. */
static void* (*old_malloc_hook)(size_t, const void*);
static void* (*old_realloc_hook)(void*, size_t, const void*);
static void* (*old_memalign_hook)(size_t, size_t, const void*);
static void (*old_free_hook)(void*, const void*);
// static void (*old_malloc_initialize_hook)(void);
// static void (*old_after_morecore_hook)(void);

/* Override initializing hook from the C library. */
void (*__malloc_initialize_hook)(void) = my_init_hook;
static void my_init_hook(void) {
    old_malloc_hook = __malloc_hook;
    old_realloc_hook = __realloc_hook;
    // old_memalign_hook = __memalign_hook;
    old_free_hook = __free_hook;
    // old_malloc_initialize_hook = __malloc_initialize_hook;
    // old_after_morecore_hook = __after_morecore_hook;

    __malloc_hook = my_malloc_hook;
    __realloc_hook = my_realloc_hook;
    // __memalign_hook = my_memalign_hook;
    __free_hook = my_free_hook;
    // __malloc_initialize_hook = my_malloc_initialize_hook;
    // __after_morecore_hook = my_after_morecore_hook;
}
static void* my_malloc_hook(size_t size, const void* caller) {
    std::lock_guard<std::mutex> lk(g_mutex);
    void* result;
    /* Restore all old hooks */
    __malloc_hook = old_malloc_hook;
    /* Call recursively */
    result = malloc(size);
    /* Save underlying hooks */
    old_malloc_hook = __malloc_hook;
    RecordMem::getInstance()->add(result, size);
#    if PRINT_LOG
    printf("-->malloc(%u) called from %p returns %p\n", (unsigned int)size, caller, result);
#    endif
    /* Restore our own hooks */
    __malloc_hook = my_malloc_hook;
    return result;
}

static void* my_realloc_hook(void* ptr, size_t size, const void* caller) {
    std::lock_guard<std::mutex> lk(g_mutex);
    void* result;
    /* Restore all old hooks */
    __realloc_hook = old_realloc_hook;
    /* Call recursively */
    result = realloc(ptr, size);
    /* Save underlying hooks */
    old_realloc_hook = __realloc_hook;
#    if PRINT_LOG
    /* printf() might call malloc(), so protect it too. */
    printf("realloc(%u) called from %p returns %p\n", (unsigned int)size, caller, result);
#    endif
    /* Restore our own hooks */
    __realloc_hook = my_realloc_hook;
    return result;
}

static void* my_memalign_hook(size_t alignment, size_t size, const void* caller) {
    std::lock_guard<std::mutex> lk(g_mutex);
    void* result;
    /* Restore all old hooks */
    __memalign_hook = old_memalign_hook;
    /* Call recursively */
    result = memalign(alignment, size);
    /* Save underlying hooks */
    old_memalign_hook = __memalign_hook;
#    if PRINT_LOG
    /* printf() might call memalign(), so protect it too. */
    printf("memalign(%u) called from %p returns %p\n", (unsigned int)size, caller, result);
#    endif
    /* Restore our own hooks */
    __memalign_hook = my_memalign_hook;
    return result;
}
static void my_free_hook(void* ptr, const void* caller) {
    std::lock_guard<std::mutex> lk(g_mutex);
    void* result;
    /* Restore all old hooks */
    __free_hook = old_free_hook;
    /* Call recursively */
    free(ptr);
    /* Save underlying hooks */
    old_free_hook = __free_hook;
    /* printf() might call free(), so protect it too. */
    RecordMem::getInstance()->remove(ptr);
#    if PRINT_LOG
    printf("<--free %p from %p\n", ptr, caller);
#    endif
    /* Restore our own hooks */
    __free_hook = my_free_hook;
}

// static void my_malloc_initialize_hook(void){
//     void *result;
//     /* Restore all old hooks */
//     __malloc_hook = old_malloc_hook;
//     /* Call recursively */
//     result = malloc(size);
//     /* Save underlying hooks */
//     old_malloc_hook = __malloc_hook;
//     /* printf() might call malloc(), so protect it too. */
//     printf("malloc(%u) called from %p returns %p\n",
//            (unsigned int)size, caller, result);
//     /* Restore our own hooks */
//     __malloc_hook = my_malloc_hook;
//     return result;
// }
// static void my_after_morecore_hook(void){
//     void *result;
//     /* Restore all old hooks */
//     __malloc_hook = old_malloc_hook;
//     /* Call recursively */
//     result = malloc(size);
//     /* Save underlying hooks */
//     old_malloc_hook = __malloc_hook;
//     /* printf() might call malloc(), so protect it too. */
//     printf("malloc(%u) called from %p returns %p\n",
//            (unsigned int)size, caller, result);
//     /* Restore our own hooks */
//     __malloc_hook = my_malloc_hook;
//     return result;
// }

#endif  // ENABLE_TRACE_MEM_USAGE