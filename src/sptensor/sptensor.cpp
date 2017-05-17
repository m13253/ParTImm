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

SparseTensor::SparseTensor(size_t nmodes, size_t const shape[], bool const is_dense[]) {

    // nmodes
    this->nmodes = nmodes;

    // shape
    this->shape.allocate(cpu, nmodes);
    std::memcpy(this->shape(cpu), shape, nmodes * sizeof (size_t));

    // is_dense
    this->is_dense.allocate(cpu, nmodes);
    std::memcpy(this->is_dense(cpu), is_dense, nmodes * sizeof (bool));

    size_t dense_modes = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        if(is_dense[m]) {
            ++dense_modes;
        }
    }

    // dense_order
    this->dense_order.allocate(cpu, dense_modes);
    size_t* dense_order = this->dense_order(cpu);
    size_t order_idx = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        if(is_dense[m]) {
            dense_order[order_idx++] = m;
        }
    }

    // sparse_order
    this->sparse_order.allocate(cpu, nmodes - dense_modes);
    size_t* sparse_order = this->sparse_order(cpu);
    order_idx = 0;
    for(size_t m = 0; m < nmodes; ++m) {
        if(!is_dense[m]) {
            sparse_order[order_idx++] = m;
        }
    }

    // strides
    this->strides.allocate(cpu, nmodes);
    size_t* strides = this->strides(cpu);
    for(size_t m = 0; m < nmodes; ++m) {
        if(is_dense[m]) {
            strides[m] = ceil_div<size_t>(shape[m], 8) * 8;
        } else {
            strides[m] = 1;
        }
    }

    // chunk_size
    this->chunk_size = 1;
    for(size_t m = 0; m < nmodes; ++m) {
        this->chunk_size *= strides[m];
    }

    // num_chunks
    this->num_chunks = 0;

    // indices
    this->indices = new MemBlock<size_t[]> [nmodes];
}

SparseTensor::~SparseTensor() {
    delete[] this->indices;
}

}
