#include "distributed/serialization.hpp"

#include <stdexcept>
#include <cstddef>
#include <cstdint>

void distributed::append_uint16(ByteBuffer &buffer, std::uint16_t value)
{
    buffer.push_back(static_cast<std::byte>((value >> 8) & 0xFF));
    buffer.push_back(static_cast<std::byte>(value & 0xFF));
}

std::uint16_t distributed::read_uint16(const ByteBuffer &buffer, std::size_t &offset)
{
    if (offset + 2 > buffer.size())
    {
        throw std::out_of_range("Buffer overflow while reading uint16_t");
    }

    std::uint16_t value = (static_cast<std::uint16_t>(std::to_integer<unsigned char>(buffer[offset])) << 8) |
                          static_cast<std::uint16_t>(std::to_integer<unsigned char>(buffer[offset + 1]));
    offset += 2;
    return value;
}
