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
#include <ParTI/error.hpp>
#include <ParTI/sptensor.hpp>

namespace pti {

SparseTensor tucker_decomposition(
    SparseTensor&   X,
    size_t const    R[],
    size_t const    dimorder[],
    double          tol,
    unsigned        maxiters
) {
    size_t N = X.nmodes;
    double normX = X.norm();

    std::unique_ptr<SparseTensor[]> U(new SparseTensor [N]);
    size_t U_shape[2];
    bool U_is_dense[2] = {true, true};
    for(size_t ni = 1; ni < N; ++ni) {
        size_t n = dimorder[ni];
        U_shape[0] = X.shape(cpu)[n];
        U_shape[1] = R[n];
        U[n].reset(2, U_shape, U_is_dense);
        //U[n].rand();
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
                    SparseTensor *Utilde_next = new SparseTensor;
                    *Utilde_next = tensor_times_matrix(*Utilde, U[m], m);
                    delete Utilde;
                    Utilde = Utilde_next;
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
