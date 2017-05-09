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
#include <memory>

namespace pti {

namespace {

int compare_indices(SparseTensor& tsr, size_t i, size_t j) {
    for(size_t m = 0; m < tsr.sparse_order.size(); ++m) {
        size_t mode = tsr.sparse_order.get(0)[m];
        size_t idx_i = tsr.indices[mode].get(0)[i];
        size_t idx_j = tsr.indices[mode].get(0)[j];
        if(idx_i < idx_j) {
            return -1;
        } else if(idx_i > idx_j) {
            return 1;
        }
    }
    return 0;
}

void swap_values(SparseTensor& tsr, size_t i, size_t j, Scalar* swap_buffer) {
    std::memcpy(swap_buffer, &tsr.values.get(0)[i * tsr.chunk_size], tsr.chunk_size * sizeof (Scalar));
    std::memcpy(&tsr.values.get(0)[i * tsr.chunk_size], &tsr.values.get(0)[j * tsr.chunk_size], tsr.chunk_size * sizeof (Scalar));
    std::memcpy(&tsr.values.get(0)[i * tsr.chunk_size], swap_buffer, tsr.chunk_size * sizeof (Scalar));
}

void quick_sort_index(SparseTensor& tsr, size_t l, size_t r, Scalar* swap_buffer) {
    size_t i, j, p;
    if(r-l < 2) {
        return;
    }
    p = (l+r) / 2;
    for(i = l, j = r-1; ; ++i, --j) {
        while(compare_indices(tsr, i, p) < 0) {
            ++i;
        }
        while(compare_indices(tsr, p, j) < 0) {
            --j;
        }
        if(i >= j) {
            break;
        }
        swap_values(tsr, i, j, swap_buffer);
        if(i == p) {
            p = j;
        } else if(j == p) {
            p = i;
        }
    }
    quick_sort_index(tsr, l, i, swap_buffer);
    quick_sort_index(tsr, i, r, swap_buffer);
}

}

void SparseTensor::sort_index() {
    return sort_index(this->sparse_order.get(0));
}

void SparseTensor::sort_index(size_t const sparse_order[]) {
    std::memcpy(this->sparse_order.get(0), sparse_order, this->sparse_order.size() * sizeof (size_t));

    std::unique_ptr<Scalar[]> swap_buffer(new Scalar [this->chunk_size]);

    quick_sort_index(*this, 0, this->num_chunks, swap_buffer.get());
}

}