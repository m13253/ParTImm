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

#ifndef PTI_SPTENSOR_INCLUDED
#define PTI_SPTENSOR_INCLUDED

#include <cstddef>
#include <cstdio>
#include <ParTI/base_tensor.hpp>
#include <ParTI/scalar.hpp>
#include <ParTI/memblock.hpp>

namespace pti {

struct SparseTensor : public BaseTensor {

    MemBlock<bool[]> is_dense;

    MemBlock<size_t[]> dense_order;

    MemBlock<size_t[]> sparse_order;

    MemBlock<size_t[]> strides;

    size_t chunk_size; // product of strides

    size_t num_chunks;

    MemBlock<size_t[]>* indices;

    MemBlock<Scalar[]> values;

public:

    SparseTensor(size_t nmodes, size_t const shape[], bool const mode_is_sparse[]);

    ~SparseTensor();

    void index_to_coord(size_t coord[], size_t index);

    void dump(std::FILE* fp, size_t start_index = 0);

    static SparseTensor load(std::FILE* fp, size_t start_index = 0);

    void append(size_t const coord[], Scalar value);

};

}

#endif
