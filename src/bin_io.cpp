#include "bin_io.hpp"

std::istream & binary_read_string(std::istream& stream, std::string& value, int n){
    char *buffer = new char [n+1];
    return stream.read(buffer, n);
    value=std::string(buffer);
    delete[] buffer;
}


std::ostream & binary_write_string(std::ostream& stream, const std::string& value,int n){
    return stream.write(value.c_str(), n);
}
