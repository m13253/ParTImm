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
#include <memory>
#include <ParTI/utils.hpp>

namespace pti {

void SparseTensor::append(size_t const coord[], Scalar value) {
    for(size_t m = 0; m < nmodes; ++m) {
        indices[m].copy_to(0);
        if(indices[m].size() == num_chunks) { // Need reallocation
            size_t new_size = indices[m].size() >= 8 ?
                indices[m].size() + indices[m].size() / 2 :
                8;
            indices[m].resize(0, new_size);
        }
    }

    values.copy_to(0);
    if(values.size() == num_chunks * chunk_size) { // Need reallocation
        size_t new_size = values.size() >= 8 ?
            values.size() + values.size() / 2 :
            8;
        values.resize(0, new_size);
    }

    if(chunk_size == 1) { // Fast code path fore pure sparse tensor
        size_t next_offset = num_chunks;
        for(size_t m = 0; m < nmodes; ++m) {
            indices[m].get(0)[next_offset] = coord[m];
        }
        values.get(0)[next_offset] = value;
        ++num_chunks;
    } else {
        throw std::logic_error("Unimplemented");
    }
}

}
