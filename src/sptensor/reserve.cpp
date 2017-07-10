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
#include <algorithm>

namespace pti {

size_t SparseTensor::reserve(size_t size) {
    size_t result = size;

    for(size_t m = 0; m < nmodes; ++m) {
        indices[m].copy_to(cpu);
        if(indices[m].size() < size) { // Need reallocation
            indices[m].resize(cpu, size);
        } else {
            result = indices[m].size();
        }
    }

    values.copy_to(cpu);
    if(values.size() < size * chunk_size) { // Need reallocation
        values.resize(cpu, size * chunk_size);
    } else if(chunk_size != 0) {
        result = values.size() / chunk_size;
    }

    return result;
}

}
