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
#include <ParTI/memnode.hpp>
#include <ParTI/session.hpp>

namespace pti {

template <typename T>
struct MemBlock {

private:

    int last_node;

    T** pointers;

public:

    explicit MemBlock() {
        int num_nodes = session.mem_nodes.size();
        pointers = new T* [num_nodes]();
        last_node = -1;
    }

    ~MemBlock() {
        int num_nodes = session.mem_nodes.size();
        for(int i = 0; i < num_nodes; ++i) {
            if(pointers[i]) {
                session.mem_nodes[i]->free(pointers[i]);
            }
        }
        delete[] pointers;
    }

    void allocate(int node) {
        // If you need to call the constructor, allocate it on CPU, then
        //     new (get(CPU_node)) T();
        // to invoke the constructor properly.
        if(!pointers[node]) {
            pointers[node] = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(sizeof (T)));
        }
        if(last_node == -1) {
            last_node = node;
        }
    }

    void free(int node) {
        if(node != last_node && pointers[node]) {
            session.mem_nodes[node]->free(pointers[node]);
            pointers[node] = NULL;
        }
    }

    void copy_to(int node) {
        if(node != last_node) {
            allocate(node);
            if(last_node != -1) {
                session.mem_nodes[last_node]->memcpy_to(pointers[node], *session.mem_nodes[node], pointers[last_node], sizeof (T));
            }
            last_node = node;
        }
    }

    T* operator() (int node) {
        copy_to(node);
        return pointers[node];
    }

    T* ptr(int node) const {
        return pointers[node];
    }

    void mark_dirty(int node) {
        last_node = node;
    }

};

template <typename T>
struct MemBlock<T[]> {

private:

    int last_node;

    T** pointers;

    size_t* sizes;

public:

    explicit MemBlock() {
        int num_nodes = session.mem_nodes.size();
        pointers = new T* [num_nodes]();
        sizes = new size_t [num_nodes]();
        last_node = -1;
    }

    ~MemBlock() {
        int num_nodes = session.mem_nodes.size();
        for(int i = 0; i < num_nodes; ++i) {
            if(pointers[i]) {
                session.mem_nodes[i]->free(pointers[i]);
            }
        }
        delete[] pointers;
        delete[] sizes;
    }

    void allocate(int node, size_t size) {
        // If you need to call the constructor, allocate it on CPU, then
        //     new (get(CPU_node)) T [size]();
        // to invoke the constructor properly.
        if(sizes[node] != size) {
            if(pointers[node]) {
                session.mem_nodes[node]->free(pointers[node]);
            }
            pointers[node] = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(size * sizeof (T)));
            sizes[node] = size;
        }
        if(last_node == -1) {
            last_node = node;
        }
    }

    void resize(int node, size_t size) {
        if(!pointers[node]) {
            pointers[node] = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(size * sizeof (T)));
            sizes[node] = size;
        } else {
            T* new_ptr = reinterpret_cast<T*>(session.mem_nodes[node]->malloc(size * sizeof (T)));
            session.mem_nodes[node]->memcpy_from(new_ptr, pointers[node], *session.mem_nodes[node], (size < sizes[node] ? size : sizes[node]) * sizeof (T));
            session.mem_nodes[node]->free(pointers[node]);
            pointers[node] = new_ptr;
            sizes[node] = size;
        }
        if(last_node == -1) {
            last_node = node;
        }
    }

    void free(int node) {
        if(node != last_node && pointers[node]) {
            session.mem_nodes[node]->free(pointers[node]);
            pointers[node] = NULL;
            sizes[node] = 0;
        }
    }

    void copy_to(int node) {
        if(node != last_node) {
            if(last_node != -1 && sizes[last_node] != 0) {
                allocate(node, sizes[last_node]);
                session.mem_nodes[last_node]->memcpy_to(pointers[node], *session.mem_nodes[node], pointers[last_node], sizes[last_node] * sizeof (T));
                sizes[node] = sizes[last_node];
            } else {
                this->free(node);
                sizes[node] = 0;
            }
            last_node = node;
        }
    }

    T* operator() (int node) {
        copy_to(node);
        return pointers[node];
    }

    T* ptr(int node) const {
        return pointers[node];
    }

    size_t size() const {
        return last_node != -1 ? sizes[last_node] : 0;
    }

    void mark_dirty(int node) {
        last_node = node;
    }


};

}

#endif
