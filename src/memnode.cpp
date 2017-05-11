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

#include <ParTI/memnode.hpp>
#include <cstring>
#include <stdexcept>
#include <ParTI/error.hpp>

namespace pti {

void CpuMemNode::memcpy_to(void* dest, MemNode& dest_node, void* src, size_t size) {
    if(dynamic_cast<CpuMemNode*>(&dest_node)) {
        std::memcpy(dest, src, size);
    } else if(CudaMemNode* cuda_dest_node = dynamic_cast<CudaMemNode*>(&dest_node)) {
#ifdef PARTI_USE_CUDA
        cuda_dest_node->memcpy_from(dest, src, *this, size);
#else
        (void) cuda_dest_node;
        ptiCheckCUDAError(true);
#endif
    } else {
        ptiCheckError(true, 1, "Unknown memory node type");
    }
}

void CpuMemNode::memcpy_from(void* dest, void* src, pti::MemNode& src_node, size_t size) {
    if(dynamic_cast<CpuMemNode*>(&src_node)) {
        std::memcpy(dest, src, size);
    } else if(CudaMemNode* cuda_src_node = dynamic_cast<CudaMemNode*>(&src_node)) {
#ifdef PARTI_USE_CUDA
        cuda_src_node->memcpy_to(dest, *this, src, size);
#else
        (void) cuda_src_node;
        ptiCheckCUDAError(true);
#endif
    } else {
        ptiCheckError(true, 1, "Unknown memory node type");
    }
}

#ifndef PARTI_USE_CUDA
CudaMemNode::CudaMemNode(int) {
    ptiCheckCUDAError(true);
}

void* CudaMemNode::malloc(size_t) {
    ptiCheckCUDAError(true);
}

void* CudaMemNode::realloc(void*, size_t) {
    ptiCheckCUDAError(true);
}

void CudaMemNode::free(void*) {
    ptiCheckCUDAError(true);
}

void CudaMemNode::memcpy_to(void*, MemNode&, void*, size_t) {
    ptiCheckCUDAError(true);
}

void CudaMemNode::memcpy_from(void*, void*, MemNode&, size_t) {
    ptiCheckCUDAError(true);
}
#endif

}
