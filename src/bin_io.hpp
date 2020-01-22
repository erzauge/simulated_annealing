#pragma once
#include <fstream>
#include <string>
/**
 * @file
 * @brief	binary read and write funtion
 *
 */


/**
 * @brief write a array in binary file
 * @details writing the n entrys of value as binary in file.
 * inspaierd by  https://baptiste-wicht.com/posts/2011/06/write-and-read-binary-files-in-c.html
 *
 * @param stream file stream
 * @param value array that gets writen to the file
 * @param n number of element that gets writen
 * @return file stream
 */
template<typename T>
std::ostream& binary_write(std::ostream& stream, const T& value,int n){
    return stream.write(reinterpret_cast<const char*>(&value), n*sizeof(T));
}

/**
 * @brief read a array from binary file
 * @details read n elements in to value of a binary file.
 * inspaierd by  https://baptiste-wicht.com/posts/2011/06/write-and-read-binary-files-in-c.html
 *
 * @param stream file stream
 * @param value arry where the content gets writen to
 * @param n number of elements that gets read
 * @return file stream
 */
template<typename T>
std::istream & binary_read(std::istream& stream, T& value, int n){
    return stream.read(reinterpret_cast<char*>(&value), n*sizeof(T));
}

std::ostream & binary_write_string(std::ostream& stream, const std::string& value,int n);

std::istream & binary_read_string(std::istream& stream, std::string& value, int n);
