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
    size_t mode = 0;
    size_t iterations = 5;
    ParamDefinition defs[] = {
        { "-m",             PARAM_SIZET, { &mode } },
        { "--mode",         PARAM_SIZET, { &mode } },
        { "-i",             PARAM_SIZET, { &iterations } },
        { "--iters",        PARAM_SIZET, { &iterations } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() != 2) {
        std::printf("Usage: %s [OPTIONS] X U\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-m, --mode\tUse specific mode for multiplication [Default: 0]\n");
        std::printf("\t-i, --iters\tUse specific iterations to measure time [Default: 5]\n");
        std::printf("\n");
        return 1;
    }

    session.print_devices();

    CFile fX(args[0], "r");
    SparseTensor X = SparseTensor::load(fX, 1);
    fX.fclose();

    CFile fU(args[1], "r");
    Tensor U = Tensor::load(fU);
    fU.fclose();

    Timer timer(cpu);
    SparseTensor Y;

    for(size_t iter = 0; iter <= iterations; ++iter) {
        timer.start();
        Y = tensor_times_matrix(X, U, mode);
        timer.stop();
        if(iter != 0) {
            timer.print_elapsed_time("CPU TTM");
        }
    }

    for(size_t iter = 0; iter <= iterations; ++iter) {
        timer.start();
        Y = tensor_times_matrix_omp(X, U, mode);
        timer.stop();
        if(iter != 0) {
            timer.print_elapsed_time("OMP TTM");
        }
    }

    return 0;
}
