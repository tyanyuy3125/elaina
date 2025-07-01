// Code taken and modified from pbrt-v4,  
// Originally licensed under the Apache License, Version 2.0.
#include "interop.h"

ELAINA_NAMESPACE_BEGIN

namespace inter {

    class NewDeleteResource : public memory_resource {
        void* do_allocate(size_t size, size_t alignment) override {
            void* ptr = nullptr;
#if defined(_MSC_VER)  // MSVC
            ptr = _aligned_malloc(size, alignment);
            if (!ptr) {
                throw std::bad_alloc();
            }
#else  // For GCC/Clang
            if (posix_memalign(&ptr, alignment, size) != 0) {
                throw std::bad_alloc();
            }
#endif
            return ptr;
        }

        void do_deallocate(void* ptr, size_t /*bytes*/, size_t /*alignment*/) override {
            if (!ptr)
                return;
#if defined(_MSC_VER)  // MSVC
            _aligned_free(ptr);
#else  // GCC/Clang
            free(ptr);
#endif
        }

        bool do_is_equal(const memory_resource& other) const noexcept override {
            return this == &other;
        }
    };

    static NewDeleteResource* ndr = nullptr;

    memory_resource* new_delete_resource() noexcept {
        if (!ndr) {
            ndr = new NewDeleteResource;
        }
        return ndr;
    }

    static memory_resource* defaultMemoryResource = new_delete_resource();

    memory_resource* set_default_resource(memory_resource* r) noexcept {
        memory_resource* orig = defaultMemoryResource;
        defaultMemoryResource = r;
        return orig;
    }

    memory_resource* get_default_resource() noexcept {
        return defaultMemoryResource;
    }

}
ELAINA_NAMESPACE_END
