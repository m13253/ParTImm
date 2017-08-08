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
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <ParTI/device.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/memblock.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/utils.hpp>

#ifdef PARTI_USE_CUDA
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#endif

namespace pti {

namespace {

bool is_matrix(
    SparseTensor& X,
    bool fortran_style = false
) {
    if(X.nmodes != 2) {
        return false;
    }
    if(X.num_chunks != 1) {
        return false;
    }
    size_t const* dense_order = X.dense_order(cpu);
    if(fortran_style) {
        if(dense_order[0] != 1 || dense_order[1] != 0) {
            return false;
        }
    } else {
        if(dense_order[0] != 0 || dense_order[1] != 1) {
            return false;
        }
    }
    return true;
}

void init_matrix(
    SparseTensor& X,
    size_t nrows,
    size_t ncols,
    bool fortran_style = true,
    bool initialize = true
) {
    size_t shape[2] = { nrows, ncols };
    bool const is_dense[2] = { true, true };
    X.reset(2, shape, is_dense);
    X.init_single_chunk(initialize);
    size_t* dense_order = X.dense_order(cpu);
    if(fortran_style) {
        dense_order[0] = 1;
        dense_order[1] = 0;
    }
}

}

void svd(
    SparseTensor& U,
    SparseTensor& S,
    SparseTensor& V,
    SparseTensor& X,
    CudaDevice&   cuda_device
) {

#ifdef PARTI_USE_CUDA

    ptiCheckError(sizeof (Scalar) != sizeof (float), ERR_BUILD_CONFIG, "Scalar != float");

    ptiCheckError(!is_matrix(X, true), ERR_SHAPE_MISMATCH, "X should be fortran style matrix");

    cusolverDnHandle_t handle = (cusolverDnHandle_t) cuda_device.GetCusolverDnHandle();
    cusolverStatus_t status;

    size_t const* X_shape = X.shape(cpu);
    size_t svd_m = X_shape[0];
    size_t svd_n = X_shape[1];
    size_t svd_lda = X.strides(cpu)[0];

    assert(svd_m >= svd_n);
    assert(svd_m >= 1);
    assert(svd_lda >= svd_m);
    assert(svd_n >= 1);

    int svd_work_size;
    status = cusolverDnSgesvd_bufferSize(
        handle,                                // handle
        svd_m,                                 // m
        svd_n,                                 // n
        &svd_work_size                         // lwork
    );
    ptiCheckError(status, ERR_CUDA_LIBRARY, "cuSOLVER error");

    init_matrix(U, svd_m, svd_m, true, true);
    init_matrix(S, 1, svd_n, false, true);
    init_matrix(V, svd_n, svd_n, false, true);
    size_t svd_ldu = U.strides(cpu)[0];
    size_t svd_ldvt = V.strides(cpu)[1];

    assert(svd_ldu >= svd_m);
    assert(svd_ldu >= 1);
    assert(svd_ldvt >= svd_n);
    assert(svd_ldvt >= 1);

    MemBlock<Scalar[]> svd_work;
    svd_work.allocate(cuda_device.mem_node, svd_work_size);
    MemBlock<Scalar[]> svd_rwork;
    svd_rwork.allocate(cuda_device.mem_node, std::min(svd_m, svd_n) - 1);
    MemBlock<int> svd_devInfo;
    svd_devInfo.allocate(cuda_device.mem_node);

    status = cusolverDnSgesvd(
        handle,                                // handle
        'A',                                   // jobu
        'A',                                   // jobvt
        svd_m,                                 // m
        svd_n,                                 // n
        X.values(cuda_device.mem_node),        // A
        svd_lda,                               // lda (lda >= max(1, m))
        S.values(cuda_device.mem_node),        // S
        U.values(cuda_device.mem_node),        // U
        svd_ldu,                               // ldu
        V.values(cuda_device.mem_node),        // VT
        svd_ldvt,                              // ldvt
        svd_work(cuda_device.mem_node),        // work
        svd_work_size,                         // lwork
        svd_rwork(cuda_device.mem_node),       // rwork
        svd_devInfo(cuda_device.mem_node)      // devInfo
    );
    ptiCheckError(status, ERR_CUDA_LIBRARY, "cuSOLVER error");

    cudaSetDevice(cuda_device.cuda_device);
    cudaDeviceSynchronize();

    int svd_devInfo_value = *svd_devInfo(cpu);
    ptiCheckError(svd_devInfo_value != 0, ERR_CUDA_LIBRARY, ("devInfo = " + std::to_string(svd_devInfo_value)).c_str());

#else

    unused_param(t);
    unused_param(n);
    unused_param(r);
    unused_param(cuda_device);
    ptiCheckError(true, ERR_BUILD_CONFIG, "CUDA not enabled");

#endif

}

}
