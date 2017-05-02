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

#include <cstring>
#include <stdexcept>
#include <ParTI/memnode.hpp>

namespace pti {

void CpuMemNode::memcpy_sync(void* dest, MemNode& dest_node, void* src, size_t size) {
    if(dynamic_cast<CpuMemNode*>(&dest_node)) {
        std::memcpy(dest, src, size);
    } else if(CudaMemNode* cuda_dest_node = dynamic_cast<CudaMemNode*>(&dest_node)) {
#ifdef PARTI_USE_CUDA
        cuda_dest_node->memcpy_sync(dest, src, *this, size);
#else
        throw std::logic_error("CUDA not enabled");
#endif
    } else {
        throw std::logic_error("Unknown memory node type");
    }
}

void CpuMemNode::memcpy_sync(void* dest, void* src, pti::MemNode& src_node, size_t size) {
    if(dynamic_cast<CpuMemNode*>(&src_node)) {
        std::memcpy(dest, src, size);
    } else if(CudaMemNode* cuda_src_node = dynamic_cast<CudaMemNode*>(&src_node)) {
#ifdef PARTI_USE_CUDA
        cuda_src_node->memcpy_sync(dest, *this, src, size);
#else
        throw std::logic_error("CUDA not enabled");
#endif
    } else {
        throw std::logic_error("Unknown memory node type");
    }
}

}
