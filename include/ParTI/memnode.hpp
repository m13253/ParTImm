/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PTI_MEMNODE_INCLUDED
#define PTI_MEMNODE_INCLUDED

#include <cstdio>
#include <new>
#include <string>
#include <unordered_map>
#include <ParTI/utils.hpp>

namespace pti {

struct MemNode {

    MemNode();
    virtual ~MemNode();

    virtual void* malloc(size_t size) = 0;
    virtual void* realloc(void* ptr, size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void memcpy_to(void* dest, MemNode& dest_node, void* src, size_t size) = 0;
    virtual void memcpy_from(void* dest, void* src, MemNode& src_node, size_t size) = 0;
    size_t bytes_allocated() const { return bytes_all_alloc; }
    size_t max_bytes_allocated() const { return max_bytes_all_alloc; }
    std::string bytes_allocated_str() const { return size_to_string(bytes_all_alloc); }
    std::string max_bytes_allocated_str() const { return size_to_string(max_bytes_all_alloc); }

protected:

    void profile(void const* ptr, size_t size) {
        if(enable_profiling) {
            std::unordered_map<void const*, size_t>::iterator it = bytes_per_alloc.find(ptr);
            if(it == bytes_per_alloc.end()) {
                if(size != 0) {
                    bytes_per_alloc.insert(std::make_pair(ptr, size));
                    bytes_all_alloc += size;
                }
            } else if(size != 0) {
                bytes_all_alloc -= it->second;
                it->second = size;
                bytes_all_alloc += size;
            } else {
                bytes_all_alloc -= it->second;
                bytes_per_alloc.erase(it);
            }
            if(bytes_all_alloc > max_bytes_all_alloc) {
                max_bytes_all_alloc = bytes_all_alloc;
            }
        }
    }

    bool enable_profiling = false;
    size_t bytes_all_alloc = 0;
    size_t max_bytes_all_alloc = 0;
    std::unordered_map<void const*, size_t> bytes_per_alloc;

};

struct CpuMemNode : public MemNode {

    void* malloc(size_t size) {
        if(size == 0) {
            size = 1;
        }
        void* ptr = std::malloc(size);
        if(!ptr) {
            std::fprintf(stderr, "[CpuMemNode] Failed to allocate %zu bytes.\n", size);
            throw std::bad_alloc();
        }
        profile(ptr, size);
        if(enable_profiling) {
            std::fprintf(stderr, "[CpuMemNode] malloc(%zu),\t%s used, %s max\n", size, bytes_allocated_str().c_str(), max_bytes_allocated_str().c_str());
        }
        return ptr;
    }

    void* realloc(void* ptr, size_t size) {
        if(size == 0) {
            size = 1;
        }
        void* newptr = std::realloc(ptr, size);
        if(!newptr) {
            std::fprintf(stderr, "[CpuMemNode] Failed to reallocate %zu bytes.\n", size);
            throw std::bad_alloc();
        }
        profile(ptr, 0);
        profile(newptr, size);
        if(enable_profiling) {
            std::fprintf(stderr, "[CpuMemNode] realloc(..., %zu),\t%s used, %s max\n", size, bytes_allocated_str().c_str(), max_bytes_allocated_str().c_str());
        }
        return newptr;
    }

    void free(void* ptr) {
        std::free(ptr);
        profile(ptr, 0);
    }

    void memcpy_to(void* dest, MemNode& dest_node, void* src, size_t size);
    void memcpy_from(void* dest, void* src, MemNode& src_node, size_t size);

};

struct CudaMemNode : public MemNode {

    explicit CudaMemNode(int cuda_device);
    void* malloc(size_t size);
    void* realloc(void* ptr, size_t size);
    void free(void* ptr);
    void memcpy_to(void* dest, MemNode& dest_node, void* src, size_t size);
    void memcpy_from(void* dest, void* src, MemNode& src_node, size_t size);

    int cuda_device;

};

// Not used
struct ClMemNode : public MemNode {

    explicit ClMemNode(void* cl_device);
    void* malloc(size_t size);
    void* realloc(void* ptr, size_t size);
    void free(void* ptr);
    void memcpy_to(void* dest, MemNode& dest_node, void* src, size_t size);
    void memcpy_from(void* dest, void* src, MemNode& src_node, size_t size);

    void* cl_device;

};

}

#endif
