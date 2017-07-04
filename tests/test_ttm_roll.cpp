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
#include <ParTI/algorithm.hpp>
#include <ParTI/argparse.hpp>
#include <ParTI/error.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>

using namespace pti;

int main(int argc, char const* argv[]) {
    bool dense_format = false;
    size_t limit = 10;
    ParamDefinition defs[] = {
        { "-d",             PARAM_BOOL,  { &dense_format } },
        { "--dense-format", PARAM_BOOL,  { &dense_format } },
        { "-l",             PARAM_SIZET, { &limit } },
        { "--limit",        PARAM_SIZET, { &limit } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() != 2 && args.size() != 3) {
        std::printf("Usage: %s [OPTIONS] X U1 U2 ...\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-d, --dense-format\tPrint tensor in dense format instead of sparse format.\n");
        std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
        std::printf("\n");
        return 1;
    }

    session.print_devices();

    int io_result;

    std::FILE* fX = std::fopen(args[0], "r");
    ptiCheckOSError(!fX);
    SparseTensor X = SparseTensor::load(fX, 1);
    io_result = std::fclose(fX);
    ptiCheckOSError(io_result != 0);
    std::printf("X = %s\n", X.to_string(!dense_format, limit).c_str());

    for(size_t argi = 1; argi < args.size(); ++argi) {

        std::FILE* fU = std::fopen(args[argi], "r");
        ptiCheckOSError(!fU);
        SparseTensor U = SparseTensor::load(fU, 1).to_fully_dense();
        io_result = std::fclose(fU);
        ptiCheckOSError(io_result != 0);
        std::printf("U[%zu] = %s\n", argi, U.to_string(false, limit).c_str());

        size_t mode = X.nmodes - argi;

        Timer timer(cpu);
        timer.start();
        SparseTensor Y = tensor_times_matrix(X, U, mode);
        timer.stop();

        std::printf("Y[%zu] = %s\n", argi, Y.to_string(!dense_format, limit).c_str());
        timer.print_elapsed_time("TTM");

        X = std::move(Y);

    }

    return 0;
}
