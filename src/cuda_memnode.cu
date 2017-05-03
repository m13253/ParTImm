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

#include <stdexcept>
#include <ParTI/memnode.hpp>

namespace pti {

CudaMemNode::CudaMemNode(int cuda_device) {
    this->cuda_device = cuda_device;
}

void* CudaMemNode::malloc(size_t size) {
    cudaError_t error;
    int old_device;
    error = cudaGetDevice(&old_device);
    if(error) { throw std::bad_alloc(); }
    error = cudaSetDevice(cuda_device);
    if(error) { throw std::bad_alloc(); }
    void* ptr;
    error = cudaMalloc(&ptr, size);
    if(error) { throw std::bad_alloc(); }
    cudaSetDevice(old_device);
    return ptr;
}

void* CudaMemNode::realloc(void*, size_t) {
    throw std::bad_alloc();
}

void CudaMemNode::free(void* ptr) {
    cudaError_t error;
    int old_device;
    error = cudaGetDevice(&old_device);
    if(error) { throw std::bad_alloc(); }
    error = cudaSetDevice(cuda_device);
    if(error) { throw std::bad_alloc(); }
    error = cudaFree(ptr);
    if(error) { throw std::bad_alloc(); }
    cudaSetDevice(old_device);
}

void CudaMemNode::memcpy_to(void* dest, MemNode& dest_node, void* src, size_t size) {
    if(CpuMemNode* cpu_dest_node = dynamic_cast<CpuMemNode*>(&dest_node)) {
        cudaSetDevice(cuda_device);
        cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    } else if(CudaMemNode* cuda_dest_node = dynamic_cast<CudaMemNode*>(&dest_node)) {
        cudaMemcpyPeer(dest, cuda_dest_node->cuda_device, src, cuda_device, size);
    } else {
        throw std::logic_error("Unknown memory node type");
    }
}

void CudaMemNode::memcpy_from(void* dest, void* src, MemNode& src_node, size_t size) {
    if(CpuMemNode* cpu_src_node = dynamic_cast<CpuMemNode*>(&src_node)) {
        cudaSetDevice(cuda_device);
        cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    } else if(CudaMemNode* cuda_src_node = dynamic_cast<CudaMemNode*>(&src_node)) {
        cudaMemcpyPeer(dest, cuda_device, src, cuda_src_node->cuda_device, size);
    } else {
        throw std::logic_error("Unknown memory node type");
    }
}


}
