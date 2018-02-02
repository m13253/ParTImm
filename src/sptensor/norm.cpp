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
#include <cmath>
#include <cstring>
#include <memory>

namespace pti {

double SparseTensor::norm() {
    double sqnorm = 0;
    Scalar *values = this->values(cpu);
    size_t *shape = this->shape(cpu);
    size_t num_dense_order = this->dense_order.size();

    if(num_dense_order == 0) {
        // Fully sparse
        for(size_t i = 0; i < num_chunks; ++i) {
            double cell_value = values[i * chunk_size];
            sqnorm += cell_value * cell_value;
        }
    } else {
        size_t *dense_order = this->dense_order(cpu);
        std::unique_ptr<size_t []> coord(new size_t [num_dense_order]);

        for(size_t i = 0; i < num_chunks; ++i) {
            std::memset(coord.get(), 0, nmodes * sizeof (size_t));
            for(;;) {
                for(size_t m = num_dense_order - 1; m != 0; --m) {
                    if(coord[m] >= shape[dense_order[m]]) {
                        coord[m] = 0;
                        ++coord[m - 1];
                    } else {
                        break;
                    }
                }
                if(coord[0] >= shape[dense_order[0]]) {
                    break;
                }
                double cell_value = values[i * chunk_size + indices_to_intra_offset(coord.get())];
                sqnorm += cell_value * cell_value;
                ++coord[num_dense_order - 1];
            }
        }
    }

    return std::sqrt(sqnorm);
}

}
