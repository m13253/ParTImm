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

#include <cstdio>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/device.hpp>

namespace pti {

void Timer::cuda_init() {
    cudaEventCreate((cudaEvent_t*) &cuda_start_event);
    cudaEventCreate((cudaEvent_t*) &cuda_stop_event);
}

void Timer::cuda_fini() {
    cudaEventDestroy((cudaEvent_t) cuda_start_event);
    cudaEventDestroy((cudaEvent_t) cuda_stop_event);
}

void Timer::cuda_start() {
    cudaSetDevice(cuda_dev->cuda_device);
    cudaEventRecord((cudaEvent_t) cuda_start_event);
    cudaEventSynchronize((cudaEvent_t) cuda_start_event);
}

void Timer::cuda_stop() {
    cudaSetDevice(cuda_dev->cuda_device);
    cudaEventRecord((cudaEvent_t) cuda_stop_event);
    cudaEventSynchronize((cudaEvent_t) cuda_stop_event);
}

double Timer::elapsed_time() const {
    float elapsed;
    if(cudaEventElapsedTime(&elapsed, (cudaEvent_t) cuda_start_event, (cudaEvent_t) cuda_stop_event) != 0) {
        return NAN;
    }
    return elapsed * 1e-3;
}

}
