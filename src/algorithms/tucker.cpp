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
#include <cmath>
#include <memory>
#include <random>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/sptensor.hpp>
#include <cusolverDn.h>

namespace pti {

namespace {

void uniform_random_fill_matrix(
    SparseTensor&   mtx
) {
    ptiCheckError(mtx.nmodes != 2, ERR_SHAPE_MISMATCH, "mtx.nmodes != 2");
    ptiCheckError(mtx.dense_order(cpu)[0] != 0, ERR_SHAPE_MISMATCH, "mtx.dense_order[0] != 0");
    ptiCheckError(mtx.dense_order(cpu)[1] != 1, ERR_SHAPE_MISMATCH, "mtx.dense_order[1] != 1");

    std::default_random_engine generator;
    std::uniform_real_distribution<Scalar> distribution(-1.0, 1.0);

    size_t nrows = mtx.shape(cpu)[0];
    size_t ncols = mtx.shape(cpu)[1];
    size_t stride = mtx.strides(cpu)[1];
    mtx.reserve(1, false);

    Scalar* values = mtx.values(cpu);
    for(size_t i = 0; i < nrows; ++i) {
        for(size_t j = 0; j < ncols; ++j) {
            values[i * stride + j] = distribution(generator);
        }
        for(size_t j = ncols; j < stride; ++j) {
            values[i * stride + j] = 0;
        }
    }
    for(size_t i = nrows * stride; i < mtx.chunk_size; ++i) {
        values[i] = 0;
    }
}

SparseTensor nvecs(
    SparseTensor& t,
    size_t        n,
    size_t        r
) {
    return SparseTensor();
}

}

SparseTensor tucker_decomposition(
    SparseTensor&   X,
    size_t const    R[],
    size_t const    dimorder[],
    double          tol,
    unsigned        maxiters
) {
    size_t N = X.nmodes;
    double normX = X.norm();

    std::unique_ptr<SparseTensor[]> U(new SparseTensor[N]);
    size_t U_shape[2];
    bool U_is_dense[2] = {true, true};
    for(size_t ni = 1; ni < N; ++ni) {
        size_t n = dimorder[ni];
        U_shape[0] = X.shape(cpu)[n];
        U_shape[1] = R[n];
        U[n].reset(2, U_shape, U_is_dense);
        uniform_random_fill_matrix(U[n]);
    }
    SparseTensor core;

    double fit = 0;
    for(unsigned iter = 0; iter < maxiters; ++iter) {
        double fitold = fit;

        SparseTensor* Utilde = &X;
        for(size_t ni = 0; ni < N; ++ni) {
            size_t n = dimorder[ni];
            for(size_t m = 0; m < N; ++m) {
                if(m != n) {
                    std::fprintf(stderr, "Iter %u, n = %zu, m = %zu\n", iter, n, m);
                    SparseTensor *Utilde_next = new SparseTensor(tensor_times_matrix(*Utilde, U[m], m));
                    if(Utilde != &X) {
                        delete Utilde;
                    }
                    Utilde = Utilde_next;
                    std::fprintf(stderr, "Utilde = %s\n", Utilde->to_string(true).c_str());
                }
            }
            // U[n] = nvecs(Utilde, n, R[n]);
        }

        core = tensor_times_matrix(*Utilde, U[dimorder[N-1]], dimorder[N-1]);

        double normresidual = std::hypot(normX, core.norm());
        fit = 1 - normresidual / normX;
        double fitchange = std::fabs(fitold - fit);

        if(iter != 0 && fitchange < tol) {
            break;
        }
    }

    return core;
}

}
