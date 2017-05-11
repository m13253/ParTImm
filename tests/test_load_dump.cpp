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
#include <ParTI/error.hpp>
#include <ParTI/sptensor.hpp>

int main(int argc, char const* argv[]) {
    if(argc != 2 && argc != 3) {
        std::fprintf(stderr, "Usage: %s input_tensor [output_tensor]\n\n", argv[0]);
        return 1;
    }

    int io_result;

    std::FILE* fi = std::fopen(argv[1], "r");
    ptiCheckOSError(!fi);
    pti::SparseTensor tsr = pti::SparseTensor::load(fi, 1);
    io_result = std::fclose(fi);
    ptiCheckOSError(io_result != 0);

    std::printf("tsr = %s\n", tsr.to_string(true, 10).c_str());

    if(argc == 3) {
        std::FILE* fo = std::fopen(argv[2], "w");
        ptiCheckOSError(!fo);
        tsr.dump(fo, 1);
        io_result = std::fclose(fo);
        ptiCheckOSError(io_result != 0);
    }

    return 0;
}
