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

#include <ParTI/sptensor.hpp>
#include <cstdio>
#include <memory>

namespace pti {

SparseTensor SparseTensor::load(std::FILE* fp, size_t start_index) {
    size_t nmodes;
    std::fscanf(fp, "%zu", &nmodes);

    std::unique_ptr<size_t[]> coordinate(new size_t [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        std::fscanf(fp, "%zu", &coordinate[m]);
    }

    std::unique_ptr<bool[]> mode_is_sparse(new bool [nmodes]);
    for(size_t m = 0; m < nmodes; ++m) {
        mode_is_sparse[m] = true;
    }

    SparseTensor tensor(nmodes, coordinate.get(), mode_is_sparse.get());

    for(;;) {
        int io_result;
        for(size_t m = 0; m < nmodes; ++m) {
            io_result = std::fscanf(fp, "%zu", &coordinate[m]);
            if(io_result != 1) break;
            coordinate[m] -= start_index;
        }
        if(io_result != 1) break;
        double value;
        io_result = std::fscanf(fp, "%lg", &value);
        if(io_result != 1) break;
        tensor.append(coordinate.get(), value);
    }

    return tensor;
}

}