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

#ifndef PTI_ALGORITHM_INCLUDED
#define PTI_ALGORITHM_INCLUDED

#include <vector>
#include <ParTI/device.hpp>
#include <ParTI/sptensor.hpp>

namespace pti {

SparseTensor tensor_times_matrix(
    SparseTensor& X,
    Tensor& U,
    size_t mode
);

SparseTensor tensor_times_matrix_omp(
    SparseTensor& X,
    Tensor& U,
    size_t mode
);

void set_semisparse_indices_by_sparse_ref(
    SparseTensor& dest,
    std::vector<size_t>& fiber_idx,
    SparseTensor& ref,
    size_t mode
);

void svd(
    Tensor& U,
    bool U_want_transpose,
    Tensor& S,
    Tensor& V,
    bool V_want_transpose,
    Tensor& X,
    CudaDevice& cuda_device
);

SparseTensor tucker_decomposition(
    SparseTensor& X,
    size_t const R[],
    size_t const dimorder[],
    CudaDevice& cuda_device,
    double tol = 1.0e-4,
    unsigned maxiters = 50
);

}

#endif
