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

#include <ParTI/timer.hpp>
#include <cstdio>
#include <time.h>
#include <ParTI/session.hpp>
#include <ParTI/device.hpp>

namespace pti {

Timer::Timer(int device) {
    this->device = device;
    cuda_dev = dynamic_cast<CudaDevice*>(session.devices[device]);
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_init();
#else
        throw std::logic_error("CUDA support not enabled");
#endif
    }
}

Timer::~Timer() {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_fini();
#else
        throw std::logic_error("CUDA support not enabled");
#endif
    }
}

void Timer::start() {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_start();
#else
        throw std::logic_error("CUDA support not enabled");
#endif
    } else {
        clock_gettime(CLOCK_MONOTONIC, &start_timespec);
    }
}

void Timer::stop() {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        cuda_stop();
#else
        throw std::logic_error("CUDA support not enabled");
#endif
    } else {
        clock_gettime(CLOCK_MONOTONIC, &stop_timespec);
    }
}

double Timer::elapsed_time() const {
    if(cuda_dev) {
#ifdef PARTI_USE_CUDA
        return cuda_elapsed_time();
#else
        throw std::logic_error("CUDA support not enabled");
#endif
    } else {
        return stop_timespec.tv_sec - start_timespec.tv_sec
            + (stop_timespec.tv_nsec - start_timespec.tv_nsec) * 1e-9;
    }
}

double Timer::print_elapsed_time(char const* name) const {
    double elapsed_time = this->elapsed_time();
    fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    return elapsed_time;
}

}
