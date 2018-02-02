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
#include <ParTI/tensor.hpp>
#include <ParTI/utils.hpp>

#ifdef PARTI_USE_CBLAS
#include <cblas.h>
#endif

#ifdef PARTI_USE_LAPACKE
#include <lapacke.h>
#endif

#ifdef PARTI_USE_CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusolverDn.h>
#endif

namespace pti {

namespace {

void init_matrix(
    Tensor& X,
    size_t nrows,
    size_t ncols,
    bool fortran_style = true,
    bool initialize = true
) {
    size_t shape[2] = { nrows, ncols };
    X.reset(2, shape, initialize);
    size_t* storage_order = X.storage_order(cpu);
    if(fortran_style) {
        storage_order[0] = 1;
        storage_order[1] = 0;
    }
}

void transpose_matrix(
    Tensor& X,
    bool do_transpose,
    bool want_fortran_style,
    Device *device
) {

    ptiCheckError(sizeof (Scalar) != sizeof (float), ERR_BUILD_CONFIG, "Scalar != float");

    ptiCheckError(X.nmodes != 2, ERR_SHAPE_MISMATCH, "X.nmodes != 2");

    size_t* storage_order = X.storage_order(cpu);
    bool currently_fortran_style;
    if(storage_order[0] == 0 && storage_order[1] == 1) {
        currently_fortran_style = false;
    } else if(storage_order[0] == 1 && storage_order[1] == 0) {
        currently_fortran_style = true;
    } else {
        ptiCheckError(true, ERR_SHAPE_MISMATCH, "X is not a matrix");
    }

    size_t* shape = X.shape(cpu);
    size_t* strides = X.strides(cpu);

    if(do_transpose != (currently_fortran_style != want_fortran_style)) {
        size_t m = shape[storage_order[0]]; // Result rows
        size_t n = shape[storage_order[1]]; // Result cols
        size_t ldm = ceil_div<size_t>(m, 8) * 8;
        size_t ldn = strides[storage_order[1]];
        MemBlock<Scalar[]> result_matrix;
        result_matrix.allocate(device->mem_node, n * ldm);

        if(CudaDevice *cuda_device = dynamic_cast<CudaDevice *>(device)) {

#ifdef PARTI_USE_CUDA

            cublasHandle_t handle = (cublasHandle_t) cuda_device->GetCublasHandle();

            cublasStatus_t status = cublasSetPointerMode(
                handle,
                CUBLAS_POINTER_MODE_HOST
            );
            ptiCheckError(status, ERR_CUDA_LIBRARY, "cuBLAS error");

            Scalar const alpha = 1;
            Scalar const beta = 0;
            status = cublasSgeam(
                handle,                             // handle
                CUBLAS_OP_T,                        // transa
                CUBLAS_OP_N,                        // transb
                m,                                  // m
                n,                                  // n
                &alpha,                             // alpha
                X.values(device->mem_node),         // A
                ldn,                                // lda
                &beta,                              // beta
                nullptr,                            // B
                ldm,                                // ldb
                result_matrix(device->mem_node),    // C
                ldm                                 // ldc
            );
            ptiCheckError(status, ERR_CUDA_LIBRARY, "cuBLAS error");

            cudaSetDevice(cuda_device->cuda_device);
            cudaDeviceSynchronize();

#else

            ptiCheckError(true, ERR_BUILD_CONFIG, "CUDA not enabled");
#endif

        } else if(dynamic_cast<CpuDevice *>(device) != nullptr) {

#ifdef PARTI_USE_OPENBLAS

            cblas_somatcopy(
                CblasColMajor,                          // CORDER
                CblasTrans,                             // CTRANS
                n,                                      // crows
                m,                                      // ccols
                1,                                      // calpha
                X.values(device->mem_node),             // a
                ldn,                                    // clda
                result_matrix(device->mem_node),        // b
                ldm                                     // cldb
            );

#else

            Scalar *result_matrix_values = result_matrix(device->mem_node);
            const Scalar *X_values = X.values(device->mem_node);
            for(size_t i = 0; i < n; ++i) {
                for(size_t j = 0; j < m; ++j) {
                    result_matrix_values[i * ldm + j] = X_values[j * ldn + i];
                }
            }

#endif

        } else {
            ptiCheckError(true, ERR_VALUE_ERROR, "Invalid device type");
        }

        X.values = std::move(result_matrix);
    }

    if(do_transpose) {
        std::swap(shape[0], shape[1]);
        std::swap(strides[0], strides[1]);
    }

    if(want_fortran_style) {
        storage_order[0] = 1;
        storage_order[1] = 0;
    } else {
        storage_order[0] = 0;
        storage_order[1] = 1;
    }

}

}

void svd(
    Tensor* U,
    bool U_want_transpose,
    Tensor& S,
    Tensor* V,
    bool V_want_transpose,
    Tensor& X,
    Device* device
) {

    ptiCheckError(sizeof (Scalar) != sizeof (float), ERR_BUILD_CONFIG, "Scalar != float");

    size_t const* X_shape = X.shape(cpu);

    bool X_transposed = X_shape[0] < X_shape[1];
    transpose_matrix(X, X_transposed, true, device);
    if(X_transposed) {
        std::swap(U, V);
        std::swap(U_want_transpose, V_want_transpose);
    }

    size_t svd_m = X_shape[0];
    size_t svd_n = X_shape[1];
    size_t svd_lda = X.strides(cpu)[0];

    assert(svd_m >= svd_n);
    assert(svd_m >= 1);
    assert(svd_lda >= svd_m);
    assert(svd_n >= 1);

    if(U != nullptr) {
        init_matrix(*U, svd_m, svd_m, true, true);
    }
    init_matrix(S, 1, svd_n, false, true);
    if(V != nullptr) {
        init_matrix(*V, svd_n, svd_n, true, true);
    }
    size_t svd_ldu = U ? U->strides(cpu)[0] : svd_m;
    size_t svd_ldvt = V ? V->strides(cpu)[0] : svd_n;

    assert(svd_ldu >= svd_m);
    assert(svd_ldu >= 1);
    assert(svd_ldvt >= svd_n);
    assert(svd_ldvt >= 1);

    if(CudaDevice *cuda_device = dynamic_cast<CudaDevice *>(device)) {

#ifdef PARTI_USE_CUDA

        cusolverDnHandle_t handle = (cusolverDnHandle_t) cuda_device->GetCusolverDnHandle();
        cusolverStatus_t status;

        int svd_work_size;
        status = cusolverDnSgesvd_bufferSize(
            handle,                                // handle
            svd_m,                                 // m
            svd_n,                                 // n
            &svd_work_size                         // lwork
        );
        ptiCheckError(status, ERR_CUDA_LIBRARY, "cuSOLVER error");

        MemBlock<Scalar[]> svd_work;
        svd_work.allocate(device->mem_node, svd_work_size);
        MemBlock<Scalar[]> svd_rwork;
        svd_rwork.allocate(device->mem_node, std::min(svd_m, svd_n) - 1);
        MemBlock<int> svd_devInfo;
        svd_devInfo.allocate(device->mem_node);

        status = cusolverDnSgesvd(
            handle,                                     // handle
            U ? 'A' : 'N',                              // jobu
            V ? 'A' : 'N',                              // jobvt
            svd_m,                                      // m
            svd_n,                                      // n
            X.values(device->mem_node),                 // A
            svd_lda,                                    // lda (lda >= max(1, m))
            S.values(device->mem_node),                 // S
            U ? U->values(device->mem_node) : nullptr,  // U
            svd_ldu,                                    // ldu
            V ? V->values(device->mem_node) : nullptr,  // VT
            svd_ldvt,                                   // ldvt
            svd_work(device->mem_node),                 // work
            svd_work_size,                              // lwork
            svd_rwork(device->mem_node),                // rwork
            svd_devInfo(device->mem_node)               // devInfo
        );
        ptiCheckError(status, ERR_CUDA_LIBRARY, "cuSOLVER error");

        cudaSetDevice(cuda_device->cuda_device);
        cudaDeviceSynchronize();

        int svd_devInfo_value = *svd_devInfo(cpu);
        ptiCheckError(svd_devInfo_value != 0, ERR_CUDA_LIBRARY, ("devInfo = " + std::to_string(svd_devInfo_value)).c_str());

#else

        ptiCheckError(true, ERR_BUILD_CONFIG, "CUDA not enabled");

#endif

    } else if(dynamic_cast<CpuDevice *>(device) != nullptr) {

#ifdef PARTI_USE_LAPACKE

        MemBlock<Scalar[]> svd_superb;
        svd_superb.allocate(device->mem_node, std::min(svd_m, svd_n) - 1);

        lapack_int status = LAPACKE_sgesvd(
            LAPACK_COL_MAJOR,                           // matrix_layout
            U ? 'A' : 'N',                              // jobu
            V ? 'A' : 'N',                              // jobvt
            svd_m,                                      // m
            svd_n,                                      // n
            X.values(device->mem_node),                 // a
            svd_lda,                                    // lda
            S.values(device->mem_node),                 // s
            U ? U->values(device->mem_node) : nullptr,  // U
            svd_ldu,                                    // ldu
            V ? V->values(device->mem_node) : nullptr,  // VT
            svd_ldvt,                                   // ldvt
            svd_superb(device->mem_node)                // superb
        );
        ptiCheckError(status, ERR_LAPACK_LIBRARY, "LAPACKE error");

#else

        ptiCheckError(true, ERR_BUILD_CONFIG, "LAPACKE not enabled");

#endif

    } else {
        ptiCheckError(true, ERR_VALUE_ERROR, "Invalid device type");
    }

    if(U != nullptr) {
        transpose_matrix(*U, U_want_transpose != X_transposed, false, device);
    }
    if(V != nullptr) {
        transpose_matrix(*V, V_want_transpose == X_transposed, false, device);
    }
}

}
