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
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <ParTI/utils.hpp>

namespace pti {

std::string SparseTensor::to_string(bool sparse_format, size_t limit) {
    std::string result = "pti::SparseTensor(\n  shape = [";
    result += array_to_string(shape(cpu), nmodes);
    result += "],\n  dense_order = [";
    result += array_to_string(dense_order(cpu), dense_order.size());
    result += "], sparse_order = [";
    result += array_to_string(sparse_order(cpu), sparse_order.size());
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
                if(is_dense(cpu)[m]) {
                    result += ':';
                } else {
                    result += std::to_string(indices[m](cpu)[i]);
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
                result += std::to_string(values(cpu)[i * chunk_size + j]);
            }
            result += ']';
        }
        if(num_chunks != 0) {
            result += '\n';
        }
    } else if(nmodes != 0) {
        std::unique_ptr<size_t[]> mode_order(new size_t [nmodes]);
        std::memcpy(mode_order.get(), sparse_order(cpu), sparse_order.size() * sizeof (size_t));
        std::memcpy(mode_order.get() + sparse_order.size(), dense_order(cpu), dense_order.size() * sizeof (size_t));

        size_t nonzero_modes = 0;
        for(size_t m = 0; m < nmodes; ++m) {
            if(shape(cpu)[mode_order[m]] != 0) {
                ++nonzero_modes;
            } else {
                break;
            }
        }
        if(nonzero_modes != nmodes) {
            ++nonzero_modes;
        }
        size_t last_mode = mode_order[nonzero_modes - 1];

        std::unique_ptr<size_t[]> coord(new size_t [nmodes] ());
        std::unique_ptr<size_t[]> next_coord(new size_t [nmodes]);
        size_t level = 0;

        for(size_t i = 0; i <= num_chunks * chunk_size;) {
            if(level != nonzero_modes) {
                if(level != 0) {
                    result += ",\n";
                }
                for(size_t m = 0; m < level + 4; ++m) {
                    result += ' ';
                }
                for(size_t m = level; m < nonzero_modes; ++m) {
                    result += '[';
                }
                level = nonzero_modes;
            } else {
                result += ", ";
            }
            offset_to_indices(next_coord.get(), i);
            bool match_next = std::memcmp(coord.get(), next_coord.get(), nmodes * sizeof (size_t)) == 0;
            if(coord[last_mode] < shape(cpu)[last_mode]) {
                if(match_next) {
                    result += std::to_string(values(cpu)[i]);
                } else {
                    result += "0";
                }
                ++coord[last_mode];
            }
            for(size_t m = 0; m + 1 < nonzero_modes; m++) {
                size_t mode = mode_order[nonzero_modes - m - 1];
                if(limit != 0 && coord[mode] >= limit) {
                    result += ", ...";
                } else if(coord[mode] < shape(cpu)[mode]) {
                    break;
                }
                --level;
                coord[mode] = 0;
                ++coord[mode_order[nonzero_modes - m - 2]];
                result += ']';
            }
            if(limit != 0 && coord[mode_order[0]] >= limit) {
                result += ", ...";
                break;
            } else if(coord[mode_order[0]] == shape(cpu)[mode_order[0]]) {
                break;
            }
            if(match_next) {
                ++i;
            }
        }
        for(size_t m = 0; m < level; ++m) {
            result += ']';
        }
        result += '\n';
    }
    result += "  }\n)";
    return result;
}

}
