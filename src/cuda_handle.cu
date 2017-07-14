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

#include <unordered_map>
#include <ParTI/device.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <cublas.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

namespace pti {

namespace {

thread_local std::unordered_map<int, cublasHandle_t> cublasHandles;
thread_local std::unordered_map<int, cusolverDnHandle_t> cusolverDnHandles;
thread_local std::unordered_map<int, cusolverSpHandle_t> cusolverSpHandles;

}

void* CudaDevice::GetCublasHandle() {
    try {
        return cublasHandles.at(cuda_device);
    } catch(std::out_of_range) {
        cudaSetDevice(cuda_device);
        cublasHandle_t handle = nullptr;
        cublasStatus_t status = cublasCreate(&handle);
        ptiCheckError(status != CUBLAS_STATUS_SUCCESS, ERR_CUDA_LIBRARY, ("cuBLAS library error code " + std::to_string(status)).c_str());
        return handle;
    }
}

void* CudaDevice::GetCusolverDnHandle() {
    try {
        return cusolverDnHandles.at(cuda_device);
    } catch(std::out_of_range) {
        cudaSetDevice(cuda_device);
        cusolverDnHandle_t handle = nullptr;
        cusolverStatus_t status = cusolverDnCreate(&handle);
        ptiCheckError(status != CUSOLVER_STATUS_SUCCESS, ERR_CUDA_LIBRARY, ("cuBLAS library error code " + std::to_string(status)).c_str());
        return handle;
    }
}

void* CudaDevice::GetCusolverSpHandle() {
    try {
        return cusolverSpHandles.at(cuda_device);
    } catch(std::out_of_range) {
        cudaSetDevice(cuda_device);
        cusolverSpHandle_t handle = nullptr;
        cusolverStatus_t status = cusolverSpCreate(&handle);
        ptiCheckError(status != CUSOLVER_STATUS_SUCCESS, ERR_CUDA_LIBRARY, ("cuBLAS library error code " + std::to_string(status)).c_str());
        return handle;
    }
}

}
