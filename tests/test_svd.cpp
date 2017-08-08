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

#include <memory>
#include <ParTI/algorithm.hpp>
#include <ParTI/argparse.hpp>
#include <ParTI/cfile.hpp>
#include <ParTI/errcode.hpp>
#include <ParTI/error.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/utils.hpp>

#ifdef PARTI_USE_CUDA
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#endif

using namespace pti;

int main(int argc, char const* argv[]) {
    size_t limit = 10;
    int device = 0;
    ParamDefinition defs[] = {
        { "-l",             PARAM_SIZET,  { &limit } },
        { "--limit",        PARAM_SIZET,  { &limit } },
        { "--dev",          PARAM_INT,    { &device } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() != 1) {
        std::printf("Usage: %s [OPTIONS] X\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
        std::printf("\t--dev\t\tCUDA device\n");
        std::printf("\n");
        return 1;
    }

    session.print_devices();
    CudaDevice* cuda_device = dynamic_cast<CudaDevice*>(session.devices[device]);

    if(cuda_device == nullptr) {
        std::fprintf(stderr, "Please specify a CUDA computing device with --dev\n\n");
        return 1;
    }

    CFile fX(args[0], "r");
    SparseTensor X = SparseTensor::load(fX, 1).to_fully_dense();
    fX.fclose();

    X.dense_order(cpu)[0] = 1;
    X.dense_order(cpu)[1] = 0;

    std::printf("X = %s\n\n", X.to_string(false, limit).c_str());

    SparseTensor U, S, V;
    svd(U, S, V, X, *cuda_device);

    return 0;
}
