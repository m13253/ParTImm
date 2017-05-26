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

int compare_indices(SparseTensor& tsr, size_t i, size_t j, size_t except) {
    size_t* sort_order = tsr.sparse_order(cpu);
    for(size_t m = 0; m < tsr.sparse_order.size(); ++m) {
        size_t mode = sort_order[m];
        if(mode == except) {
            continue;
        }
        size_t idx_i = tsr.indices[mode](cpu)[i];
        size_t idx_j = tsr.indices[mode](cpu)[j];
        if(idx_i < idx_j) {
            return -1;
        } else if(idx_i > idx_j) {
            return 1;
        }
    }
    return 0;
}

}

void set_semisparse_indices_by_sparse_ref(SparseTensor& dest, std::vector<size_t>& fiber_idx, SparseTensor& ref, size_t mode) {
    size_t lastidx = ref.num_chunks;
    assert(dest.nmodes == ref.nmodes);

    fiber_idx.clear();
    dest.num_chunks = 0;
    std::unique_ptr<size_t[]> indices(new size_t [ref.nmodes]);
    std::unique_ptr<Scalar[]> chunk(new Scalar [dest.chunk_size] ());
    for(size_t i = 0; i < ref.num_chunks; ++i) {
        if(lastidx == ref.num_chunks || compare_indices(ref, lastidx, i, mode) != 0) {
            ref.offset_to_indices(indices.get(), i * ref.chunk_size);
            dest.append(indices.get(), chunk.get());
            lastidx = i;
            fiber_idx.push_back(i);
        }
    }
    fiber_idx.push_back(ref.num_chunks);
}

}
