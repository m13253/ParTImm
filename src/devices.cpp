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

#include <ParTI/devices.hpp>

namespace pti {

Devices::~Devices() {
}

size_t Devices::add_memory_node(std::shared_ptr<MemNode> node) {
    memory_nodes.push_back(std::move(node));
    return memory_nodes.size() - 1;
}

MemNode* Devices::get_memory_node(size_t device_index) {
    return memory_nodes[device_index].get();
}

}
