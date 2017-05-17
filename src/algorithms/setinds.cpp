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

#include <ParTI/algorithm.hpp>
#include <cassert>
#include <memory>
#include <vector>
#include <ParTI/error.hpp>
#include <ParTI/sptensor.hpp>

namespace pti {

namespace {

int compare_indices(SparseTensor& tsr, size_t i, size_t j) {
    for(size_t m = 0; m < tsr.sparse_order.size(); ++m) {
        size_t mode = tsr.sparse_order(0)[m];
        size_t idx_i = tsr.indices[mode](0)[i];
        size_t idx_j = tsr.indices[mode](0)[j];
        if(idx_i < idx_j) {
            return -1;
        } else if(idx_i > idx_j) {
            return 1;
        }
    }
    return 0;
}

}

void set_semisparse_indices_by_sparse_ref(SparseTensor& dest, std::vector<size_t>& fiber_idx, SparseTensor& ref) {
    size_t lastidx = ref.num_chunks;
    assert(dest.nmodes == ref.nmodes);

    std::unique_ptr<size_t[]> sort_order(new size_t [ref.nmodes]);
    sort_order[ref.nmodes - 1] = dest.dense_order(0)[0];
    for(size_t m = 1; m < ref.nmodes - 1; ++m) {
        if(m < dest.dense_order(0)[0]) {
            sort_order[m] = m;
        } else {
            sort_order[m] = m + 1;
        }
    }
    ref.sort_index(sort_order.get());

    fiber_idx.clear();
    dest.num_chunks = 0;
    std::unique_ptr<size_t[]> indices(new size_t [ref.nmodes]);
    for(size_t i = 0; i < ref.num_chunks; ++i) {
        if(lastidx == ref.num_chunks || compare_indices(ref, lastidx, i) != 0) {
            ref.offset_to_indices(indices.get(), i * ref.chunk_size);
            dest.append(indices.get(), 0);
        }
        lastidx = i;
        fiber_idx.push_back(i);
    }
    fiber_idx.push_back(ref.num_chunks);
}

}
