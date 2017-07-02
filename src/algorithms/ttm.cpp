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
#include <utility>
#include <vector>
#include <ParTI/error.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/utils.hpp>

namespace pti {

SparseTensor tensor_times_matrix(SparseTensor& X, SparseTensor& U, size_t mode) {

    size_t nmodes = X.nmodes;
    size_t nrows = U.shape(cpu)[0];
    size_t ncols = U.shape(cpu)[1];
    size_t stride = U.strides(cpu)[1];

    ptiCheckError(mode >= nmodes, ERR_SHAPE_MISMATCH, "shape mismatch");

    ptiCheckError(U.nmodes != 2, ERR_SHAPE_MISMATCH, "U.nmodes != 2");
    ptiCheckError(U.dense_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "U.dense_order[0] != 0");
    ptiCheckError(U.dense_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "U.dense_order[1] != 1");
    ptiCheckError(X.shape(cpu)[mode] != nrows, ERR_SHAPE_MISMATCH, "X.shape[mode] != U.nrows");

    ptiCheckError(X.is_dense(cpu)[mode], ERR_UNKNOWN, "fixme: X.is_dense[mode] should be false");

    std::unique_ptr<size_t[]> sort_order(new size_t [nmodes]);
    sort_order[nmodes - 1] = mode;
    for(size_t m = 0; m < nmodes - 1; ++m) {
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
    bool const* X_is_dense = X.is_dense(cpu);
    std::unique_ptr<bool[]> Y_is_dense(new bool [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        Y_is_dense[m] = X_is_dense[m] || m == mode;
    }

    SparseTensor Y(nmodes, Y_shape.get(), Y_is_dense.get());
    Y.sort_index(sort_order.get());

    std::vector<size_t> fiberidx;
    set_semisparse_indices_by_sparse_ref(Y, fiberidx, X, mode);

    Scalar* X_values = X.values(cpu);
    Scalar* Y_values = Y.values(cpu);
    Scalar* U_values = U.values(cpu);

    std::fprintf(stderr, "X = %s\n", X.to_string(true).c_str());
    std::fprintf(stderr, "U = %s\n", U.to_string(true).c_str());
    std::fprintf(stderr, "fiberidx = %s\n", array_to_string(fiberidx.data(), fiberidx.size()).c_str());

    Timer timer(cpu);
    timer.start();

    size_t Y_slice_size = Y.strides(cpu)[mode];
    // Y_num_slices should == X.chunk_size
    size_t Y_num_slices = Y.chunk_size / Y_slice_size;
    // i is chunk-level on Y
    for(size_t i = 0; i < Y.num_chunks; ++i) {
        size_t inz_begin = fiberidx[i];
        size_t inz_end = fiberidx[i + 1];
        // j is chunk-level on X,
        // for each Y[i] corresponds to all X[j]
        for(size_t j = inz_begin; j < inz_end; ++j) {
            size_t r = X.indices[mode](cpu)[j];
            // We will cut a chunk on Y into slices * fibers,
            // a slice on Y corresponds to a chunk on X
            for(size_t k = 0; k < Y_num_slices; ++k) {
                // Then we iterate columns from U
                for(size_t c = 0; c < ncols; ++c) {
                    Y_values[i * Y.chunk_size + k * Y_slice_size + c] += X_values[j * X.chunk_size + k] * U_values[r * stride + c];
                }
            }
        }
    }

    timer.stop();
    timer.print_elapsed_time("CPU TTM");

    return Y;
}

}
