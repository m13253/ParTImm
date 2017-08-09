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

#include <ParTI/tensor.hpp>
#include <cstring>
#include <ParTI/utils.hpp>

namespace pti {

bool Tensor::offset_to_indices(size_t indices[], size_t offset) {
    if(offset >= chunk_size) {
        std::memcpy(indices, this->shape(cpu), nmodes * sizeof (size_t));
        return false;
    }
    if(chunk_size != 1) { // Semi sparse tensor
        size_t intra_chunk = offset % chunk_size;
        size_t* storage_order = this->storage_order(cpu);
        size_t* strides = this->strides(cpu);
        for(size_t o = this->storage_order.size()-1; o != 0; --o) {
            size_t m = storage_order[o];
            indices[m] = intra_chunk % strides[m];
            intra_chunk /= strides[m];
        }
        indices[storage_order[0]] = intra_chunk;
    }
    bool inbound = true;
    for(size_t m = 0; m < this->nmodes; ++m) {
        inbound = inbound && indices[m] < this->shape(cpu)[m];
    }
    return inbound;
}

}
