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

#include <cstdlib>
#include <new>

namespace pti {

struct MemNode {

    virtual void* malloc(size_t size) = 0;
    virtual void* realloc(void* ptr, size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void memcpy_sync(void* dest, MemNode& dest_node, void* src, size_t size) = 0;
    virtual void memcpy_sync(void* dest, void* src, MemNode& src_node, size_t size) = 0;

};

struct CpuMemNode : public MemNode {

    void* malloc(size_t size) {
        if(size == 0) {
            size = 1;
        }
        void* ptr = std::malloc(size);
        if(!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void* realloc(void* ptr, size_t size) {
        if(size == 0) {
            size = 1;
        }
        ptr = std::realloc(ptr, size);
        if(!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void free(void* ptr) {
        std::free(ptr);
    }

    void memcpy_sync(void* dest, MemNode& dest_node, void* src, size_t size);
    void memcpy_sync(void* dest, void* src, MemNode& src_node, size_t size);

};

struct CudaMemNode : public MemNode {

    explicit CudaMemNode(int cuda_device);
    void* malloc(size_t size);
    void* realloc(void* ptr, size_t size);
    void free(void* ptr);
    void memcpy_sync(void* dest, MemNode& dest_node, void* src, size_t size);
    void memcpy_sync(void* dest, void* src, MemNode& src_node, size_t size);

    int cuda_device;

};

struct ClMemNode : public MemNode {

    explicit ClMemNode(void* cl_device);
    void* malloc(size_t size);
    void* realloc(void* ptr, size_t size);
    void free(void* ptr);
    void memcpy_sync(void* dest, MemNode& dest_node, void* src, size_t size);
    void memcpy_sync(void* dest, void* src, MemNode& src_node, size_t size);

    void* cl_device;

};

}

#endif
