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

#ifndef PTI_MEMBLOCK_INCLUDED
#define PTI_MEMBLOCK_INCLUDED

#include <cstddef>

namespace pti {

template <typename T>
struct MemBlock {

private:

    size_t num_nodes;

    size_t last_node;

    T** pointers;

public:

    explicit MemBlock(size_t num_nodes) {
        this->num_nodes = num_nodes;
        this->last_node = (size_t) -1;
        pointers = new T* [num_nodes];
    }

    ~MemBlock() {
        delete[] pointers;
    }

    void copy_to(size_t node) {
        if(node != last_node) {
            // Do copy
        }
    }

    T* get(size_t node) {
        copy_to(node);
        return pointers[node];
    }

};

}

#endif
