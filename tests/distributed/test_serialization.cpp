#include "distributed/serialization.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

int main()
{
    distributed::ByteBuffer buffer;

    distributed::append_uint16(buffer, 0x1234U);

    assert(buffer.size() == 2);
    assert(std::to_integer<unsigned int>(buffer[0]) == 0x12U);
    assert(std::to_integer<unsigned int>(buffer[1]) == 0x34U);

    std::size_t offset = 0;

    const std::uint16_t decoded =
        distributed::read_uint16(buffer, offset);

    assert(decoded == 0x1234U);
    assert(offset == 2);

    std::cout << "serialization tests passed\n";
}
