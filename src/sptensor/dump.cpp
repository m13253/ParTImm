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
#include <ParTI/utils.hpp>
#include <cstdio>
#include <memory>

namespace pti {

void SparseTensor::dump(std::FILE* fp, size_t start_index) {
    std::fprintf(fp, "%zu\n", nmodes);
    std::fprintf(fp, "%s\n", array_to_string(shape.get(0), shape.size(), "\t").c_str());

    std::unique_ptr<size_t[]> coordinate(new size_t [nmodes]);
    for(size_t i = 0; i < num_chunks * chunk_size; ++i) {
        offset_to_indices(coordinate.get(), i);
        bool out_of_range = false;
        for(size_t m = 0; m < nmodes; ++m) {
            if(coordinate[m] > shape.get(0)[m]) {
                out_of_range = true;
                break;
            }
            coordinate[m] += start_index;
        }
        if(!out_of_range) {
            std::fprintf(fp, "%s\t%.16lg\n",
                array_to_string(coordinate.get(), nmodes, "\t").c_str(),
                (double) values.get(0)[i]);
        }
    }
}

}
