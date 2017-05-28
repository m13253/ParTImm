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

namespace pti {

double SparseTensor::norm() {
    double sqnorm = 0;
    Scalar* values = this->values(cpu);
    for(size_t i = 0; i < num_chunks * chunk_size; ++i) {
        double cell_value = values[i];
        sqnorm += cell_value * cell_value;
    }
    return std::sqrt(sqnorm);
}

}
