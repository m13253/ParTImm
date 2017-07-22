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
#include <cuda_runtime_api.h>
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
    size_t        r,
    CudaDevice&   cuda_device
) {
    ptiCheckError(sizeof (Scalar) != sizeof (float), ERR_BUILD_CONFIG, "Scalar != float");
    ptiCheckError(t.is_dense(cpu)[n] != false, ERR_SHAPE_MISMATCH, "t.is_dense[n] != false");

    cusolverDnHandle_t handle = (cusolverDnHandle_t) cuda_device.GetCusolverDnHandle();
    cusolverStatus_t status;

    bool tm_trans;
    size_t tm_m; // rows
    size_t tm_n; // colums
    if(t.shape(cpu)[n] >= t.chunk_size) {
        tm_trans = false;
        tm_m = t.shape(cpu)[n];
        tm_n = t.chunk_size;
    } else {
        tm_trans = true;
        tm_m = t.chunk_size;
        tm_n = t.shape(cpu)[n];
    }

    // cuSOLVER use FORTRAN style, swap them
    size_t tm_shape[2] = {tm_n, tm_m};
    bool full_dense[2] = {true, true};
    SparseTensor tm(2, tm_shape, full_dense);
    size_t tm_stride = tm.strides(cpu)[1];
    tm.reserve(1);

    if(!tm_trans) {
        for(size_t i = 0; i < tm_m; ++i) {
            size_t row = t.indices[n](cpu)[i];
            for(size_t j = 0; j < tm_n; ++j) {
                tm.values(cpu)[j * tm_stride + row] = t.values(cpu)[i * tm.chunk_size + j];
            }
        }
    } else {
        for(size_t i = 0; i < tm_n; ++i) {
            size_t row = t.indices[n](cpu)[i];
            for(size_t j = 0; j < tm_m; ++j) {
                tm.values(cpu)[row * tm_stride + j] = t.values(cpu)[i * tm.chunk_size + j];
            }
        }
    }

    size_t svd_m = tm_stride;
    size_t svd_n = tm_shape[0];
    size_t svd_ld = tm_shape[1];

    int svd_work_size;
    status = cusolverDnSgesvd_bufferSize(
        handle,                                // handle
        svd_m,                                 // m
        svd_n,                                 // n
        &svd_work_size                         // lwork
    );
    ptiCheckError(status, ERR_CUDA_LIBRARY, "cuSOLVER error");

    MemBlock<Scalar[]> S;
    S.allocate(cuda_device.mem_node, std::min(svd_m, svd_n));
    MemBlock<Scalar[]> U;
    U.allocate(cuda_device.mem_node, svd_m);
    MemBlock<Scalar[]> VT;
    VT.allocate(cuda_device.mem_node, svd_n);
    MemBlock<Scalar[]> svd_work;
    svd_work.allocate(cuda_device.mem_node, svd_work_size);
    MemBlock<Scalar[]> svd_rwork;
    svd_rwork.allocate(cuda_device.mem_node, std::min(svd_m, svd_n) - 1);
    MemBlock<int> svd_devInfo;
    svd_devInfo.allocate(cuda_device.mem_node);

    assert(svd_m >= svd_n);
    assert(svd_m >= 1);
    assert(svd_ld >= svd_m);
    assert(svd_n >= 1);

    status = cusolverDnSgesvd(
        handle,                                // handle
        !tm_trans ? 'O' : 'N',                 // jobu
        !tm_trans ? 'N' : 'O',                 // jobvt
        svd_m,                                 // m
        svd_n,                                 // n
        tm.values(cuda_device.mem_node),       // A
        svd_ld,                                // lda (lda >= max(1, m))
        S(cuda_device.mem_node),               // S
        U(cuda_device.mem_node),               // U
        svd_ld,                                // ldu
        VT(cuda_device.mem_node),              // VT
        svd_n,                                 // ldvt
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

    size_t result_shape[2] = {t.shape(cpu)[n], r};
    SparseTensor result(2, result_shape, full_dense);
    size_t result_stride = result.strides(cpu)[1];
    result.reserve(1);

    if(!tm_trans) {
        for(size_t i = 0; i < t.shape(cpu)[n]; ++i) {
            for(size_t j = 0; j < r; ++j) {
                result.values(cpu)[i * result_stride + j] = tm.values(cpu)[j * tm_stride + i];
            }
        }
    } else {
        for(size_t i = 0; i < t.shape(cpu)[n]; ++i) {
            for(size_t j = 0; j < r; ++j) {
                result.values(cpu)[i * result_stride + j] = tm.values(cpu)[j * tm_stride + i];
            }
        }
    }

    return result;
}

}

SparseTensor tucker_decomposition(
    SparseTensor&   X,
    size_t const    R[],
    size_t const    dimorder[],
    CudaDevice&     cuda_device,
    double          tol,
    unsigned        maxiters
) {
    ptiCheckError(X.dense_order.size() != 0, ERR_SHAPE_MISMATCH, "X should be fully sparse");

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
        SparseTensor Utilde_next;
        for(size_t ni = 0; ni < N; ++ni) {
            size_t n = dimorder[ni];
            Utilde = &X;
            for(size_t m = 0; m < N; ++m) {
                if(m != n) {
                    std::fprintf(stderr, "Iter %u, n = %zu, m = %zu\n", iter, n, m);
                    Utilde_next = tensor_times_matrix(*Utilde, U[m], m);
                    Utilde = &Utilde_next;
                    std::fprintf(stderr, "Utilde = %s\n", Utilde->to_string(true).c_str());
                }
            }
            // Mode n is sparse, while other modes are dense
            U[n] = nvecs(*Utilde, n, R[n], cuda_device);
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
