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

#include <ParTI/sptensor.hpp>
#include <cstring>
#include <ParTI/utils.hpp>

namespace pti {

void SparseTensor::offset_to_indices(size_t indices[], size_t offset) {
    bool* is_dense = this->is_dense.get(0);
    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense[m]) {
            indices[m] = this->indices[m].get(0)[offset / chunk_size];
        }
    }
    if(chunk_size != 1) { // Semi sparse tensor
        size_t intra_chunk = offset % chunk_size;
        size_t* dense_order = this->dense_order.get(0);
        size_t* strides = this->strides.get(0);
        for(size_t o = this->dense_order.size()-1; o != 0; --o) {
            size_t m = dense_order[o];
            indices[m] = intra_chunk % strides[m];
            intra_chunk /= strides[m];
        }
        indices[dense_order[0]] = intra_chunk;
    }
}

}
