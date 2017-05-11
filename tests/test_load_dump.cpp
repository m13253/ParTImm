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
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>

int main(int argc, char const* argv[]) {
    if(argc != 3) {
        std::fprintf(stderr, "Usage: %s input_tensor output_tensor\n\n", argv[0]);
        return 1;
    }

    std::FILE* fi = std::fopen(argv[1], "r");
    pti::SparseTensor tsr = pti::SparseTensor::load(fi, 1);
    std::fclose(fi);

    std::FILE* fo = std::fopen(argv[2], "w");
    tsr.dump(fo, 1);
    std::fclose(fo);
    return 0;
}
