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
#include <ParTI/errcode.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/timer.hpp>

namespace pti {

SparseTensor tensor_times_matrix(SparseTensor& X, SparseTensor& U, size_t mode) {

    size_t nmodes = X.nmodes;
    size_t nrows = U.shape(cpu)[0];
    size_t ncols = U.shape(cpu)[1];

    ptiCheckError(mode >= nmodes, ERR_SHAPE_MISMATCH, "shape mismatch");

    ptiCheckError(U.nmodes != 2, ERR_SHAPE_MISMATCH, "U.nmodes != 2");
    ptiCheckError(U.dense_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "U.dense_order[0] != 0");
    ptiCheckError(U.dense_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "U.dense_order[1] != 1");
    ptiCheckError(X.shape(cpu)[mode] != nrows, ERR_SHAPE_MISMATCH, "X.shape[mode] != U.nrows");

    std::unique_ptr<size_t[]> sort_order(new size_t [nmodes]);
    sort_order[nmodes - 1] = mode;
    for(size_t m = 1; m < nmodes - 1; ++m) {
        if(m < mode) {
            sort_order[m] = m;
        } else {
            sort_order[m] = m + 1;
        }
    }
    X.sort_index(sort_order.get());

    std::unique_ptr<size_t[]> Y_shape(new size_t [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        if(m != mode) {
            Y_shape[m] = X.shape(cpu)[m];
        } else {
            Y_shape[m] = ncols;
        }
    }
    std::unique_ptr<bool[]> Y_is_sparse(new bool [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        if(m != mode) {
            Y_is_sparse[m] = true;
        } else {
            Y_is_sparse[m] = false;
        }
    }

    SparseTensor Y(nmodes, Y_shape.get(), Y_is_sparse.get());

    std::vector<size_t> fiberidx;
    set_semisparse_indices_by_sparse_ref(Y, fiberidx, X);

    Scalar* X_values = X.values(cpu);
    Scalar* Y_values = Y.values(cpu);
    Scalar* U_values = U.values(cpu);

    Timer timer(cpu);
    timer.start();

    for(size_t i = 0; i < Y.num_chunks; ++i) {
        size_t inz_begin = fiberidx[i];
        size_t inz_end = fiberidx[i + 1];
        for(size_t j = inz_begin; j < inz_end; ++j) {
            size_t r = X.indices[mode](cpu)[j];
            for(size_t k = 0; k < ncols; ++k) {
                Y_values[i * Y.chunk_size + k] += X_values[j] * U_values[r * U.chunk_size + k];
            }
        }
    }

    timer.stop();
    timer.print_elapsed_time("CPU TTM");

    return Y;
}

}
