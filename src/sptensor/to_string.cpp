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
#include <algorithm>
#include <string>

namespace pti {

std::string SparseTensor::to_string(bool sparse_format, size_t limit) {
    std::string result = "pti::SparseTensor(\n  shape = [";
    result += array_to_string(shape.get(0), nmodes);
    result += "],\n  dense_order = [";
    result += array_to_string(dense_order.get(0), dense_order.size());
    result += "], sparse_order = [";
    result += array_to_string(sparse_order.get(0), sparse_order.size());
    result += "],\n  values[";
    result += std::to_string(num_chunks);
    result += 'x';
    result += std::to_string(chunk_size);
    result += "] = {\n";
    if(sparse_format) {
        for(size_t i = 0; i < num_chunks; ++i) {
            if(limit != 0 && i >= limit) {
                result += ",\n    ...";
                break;
            }
            if(i != 0) {
                result += ",\n";
            }
            result += "    (";
            for(size_t m = 0; m < nmodes; ++m) {
                if(m != 0) {
                    result += ", ";
                }
                if(is_dense.get(0)[m]) {
                    result += ':';
                } else {
                    result += std::to_string(indices[m].get(0)[i]);
                }
            }
            result += "): [";
            for(size_t j = 0; j < chunk_size; ++j) {
                if(limit != 0 && j >= limit) {
                    result += ", ...";
                    break;
                }
                if(j != 0) {
                    result += ", ";
                }
                result += std::to_string(values.get(0)[i * chunk_size + j]);
            }
            result += ']';
        }
        result += '\n';
    } else {
        throw std::logic_error("Unimplemented");
    }
    result += "  }\n)";
    return result;
}

}
