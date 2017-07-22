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

#ifndef PTI_UTILS_INCLUDED
#define PTI_UTILS_INCLUDED

#include <cstring>
#include <string>
#include <stdexcept>

namespace pti {

template <typename T>
inline void unused_param(T&&) {
}

template <typename T>
inline T ceil_div(T const num, T const deno) {
    return num ? (num - 1) / deno + 1 : 0;
}

template <typename T>
inline std::string array_to_string(T const array[], size_t length, std::string const& delim = ", ") {
    if(length == 0) {
        return std::string();
    }
    std::string result = std::to_string(array[0]);
    for(size_t i = 1; i < length; ++i) {
        result += delim;
        result += std::to_string(array[i]);
    }
    return result;
}

class StrToNumError : public std::runtime_error {
public:
    StrToNumError() : std::runtime_error("Invalid number format") {}
};

template <typename Fn, typename ...Args>
static inline auto strtonum(Fn fn, const char *str, Args &&...args) -> decltype(fn(str, nullptr, std::forward<Args>(args)...)) {
    if(str[0] != '\0') {
        char *endptr;
        auto result = fn(str, &endptr, std::forward<Args>(args)...);
        if(endptr == &str[std::strlen(str)]) {
            return result;
        } else {
            throw StrToNumError();
        }
    } else {
        throw StrToNumError();
    }
}

}

#endif
