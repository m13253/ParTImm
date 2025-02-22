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

#include <ParTI/algorithm.hpp>
#include <ParTI/argparse.hpp>
#include <ParTI/cfile.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/tensor.hpp>

using namespace pti;

int main(int argc, char const* argv[]) {
    int preheat = 2, count = 10;
    int device = 0;
    size_t mode = 0;
    ParamDefinition defs[] = {
        { "-p",             PARAM_INT,    { &preheat } },
        { "--preheat",      PARAM_INT,    { &preheat } },
        { "-c",             PARAM_INT,    { &count } },
        { "--count",        PARAM_INT,    { &count} },
        { "--dev",          PARAM_INT,    { &device } },
        { "--mode",         PARAM_SIZET,  { &mode } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() != 2) {
        std::printf("Usage: %s [OPTIONS] X U\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-p, --preheat\tNumber of preheat calculations [Default: 2].\n");
        std::printf("\t-c, --count\tNumber of calculations [Default: 10].\n");
        std::printf("\t-m, --mode\tUse specific mode for multiplication [Default: 0]\n");
        std::printf("\t--dev\t\tComputing device\n");
        std::printf("\n");
        return 1;
    }

    Device* dev = session.devices[device];
    if(dynamic_cast<CudaDevice*>(dev) != nullptr) {
        std::printf("Using CUDA for calculation.\n");
    } else if(dynamic_cast<CpuDevice*>(dev) != nullptr) {
        std::printf("Using CPU for calculation.\n");
    } else {
        std::printf("Unknown device type.\n");
        return 1;
    }

    CFile fX(args[0], "r");
    SparseTensor X = SparseTensor::load(fX, 1);
    fX.fclose();

    CFile fU(args[1], "r");
    Tensor U = Tensor::load(fU);
    fU.fclose();

    std::printf("Preheating...\n");
    std::fflush(stdout);

    Timer timer_single(device);
    for(int i = 0; i < std::max(preheat, 1); ++i) {
        timer_single.start();
        SparseTensor Y = tensor_times_matrix(X, U, mode, session.devices[device], i != 0);
        timer_single.stop();
        timer_single.print_elapsed_time("TTM");
    }

    std::printf("\nCalculating...\n");
    std::fflush(stdout);

    Timer timer(device);
    timer.start();
    for(int i = 0; i < count; ++i) {
        timer_single.start();
        SparseTensor Y = tensor_times_matrix(X, U, mode, session.devices[device], true);
        timer_single.stop();
        timer_single.print_elapsed_time("TTM");
    }
    timer.stop();
    std::printf("\nAverage time: %.9lf s\n", timer.elapsed_time() / count);

    return 0;
}
