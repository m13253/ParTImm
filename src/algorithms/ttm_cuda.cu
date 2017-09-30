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
#include <ParTI/tensor.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/utils.hpp>

namespace pti {

SparseTensor tensor_times_matrix_cuda(SparseTensor& X, Tensor& U, size_t mode, CudaDevice& cuda_dev) {
    throw "placeholder";
}

}
