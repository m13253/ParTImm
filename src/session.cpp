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

#include <ParTI/session.hpp>
#include <cstdio>
#include <thread>
#include <ParTI/device.hpp>
#include <ParTI/memnode.hpp>

namespace pti {

Session::Session() {
    detect_devices();
}

Session::~Session() {
    for(Device* device : devices) {
        delete device;
    }
    for(MemNode* mem_node : mem_nodes) {
        delete mem_node;
    }
}

void Session::detect_devices() {
    unsigned num_cpu_cores = std::thread::hardware_concurrency();
    CpuMemNode* cpu_mem_node = new CpuMemNode;
    int cpu_mem_node_id = add_mem_node(cpu_mem_node);
    for(unsigned i = 0; i < num_cpu_cores; ++i) {
        CpuDevice* cpu_device = new CpuDevice(i, cpu_mem_node_id);
        add_device(cpu_device);
    }
#ifdef PARTI_USE_CUDA
    detect_cuda_devices();
#endif
}

void Session::print_devices() const {
    size_t num_devices = devices.size();
    std::fprintf(stderr, "\x1b[7m  ID  Device name                                                  Memory node  \x1b[27m\n");
    for(size_t i = 0; i < num_devices; ++i) {
        Device const& device = *devices[i];
        std::fprintf(stderr, "%4zu  %-67s %4d\n", i, device.name.c_str(), device.mem_node);
    }
    if(num_devices != 1) {
        std::fprintf(stderr, "\x1b[1m%zu devices detected.\x1b[21m\n", num_devices);
    } else {
        std::fprintf(stderr, "\x1b[1m%zu device detected.\x1b[21m\n", num_devices);
    }
}

int Session::add_device(Device* device) {
    devices.push_back(device);
    return (int) devices.size() - 1;
}

int Session::add_mem_node(pti::MemNode *mem_node) {
    mem_nodes.push_back(mem_node);
    return (int) mem_nodes.size() - 1;
}

Session session;

}
