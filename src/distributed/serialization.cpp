#include "distributed/serialization.hpp"

#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cstring>

void distributed::append_uint16(ByteBuffer &buffer, std::uint16_t value)
{
    buffer.push_back(static_cast<std::byte>((value >> 8) & 0xFF));
    buffer.push_back(static_cast<std::byte>(value & 0xFF));
}

void distributed::append_uint32(ByteBuffer &buffer, std::uint32_t value)
{
    buffer.push_back(static_cast<std::byte>((value >> 24) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 16) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 8) & 0xFF));
    buffer.push_back(static_cast<std::byte>(value & 0xFF));
}

void distributed::append_uint64(ByteBuffer &buffer, std::uint64_t value)
{
    buffer.push_back(static_cast<std::byte>((value >> 56) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 48) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 40) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 32) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 24) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 16) & 0xFF));
    buffer.push_back(static_cast<std::byte>((value >> 8) & 0xFF));
    buffer.push_back(static_cast<std::byte>(value & 0xFF));
}

void distributed::append_float(ByteBuffer &buffer, float value)
{
    static_assert(sizeof(float) == 4, "Unexpected float size");
    std::uint32_t as_int;
    std::memcpy(&as_int, &value, sizeof(float));
    append_uint32(buffer, as_int);
}

void distributed::append_double(ByteBuffer &buffer, double value)
{
    static_assert(sizeof(double) == 8, "Unexpected double size");
    std::uint64_t as_int;
    std::memcpy(&as_int, &value, sizeof(double));
    append_uint64(buffer, as_int);
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

std::uint32_t distributed::read_uint32(const ByteBuffer &buffer, std::size_t &offset)
{
    if (offset + 4 > buffer.size())
    {
        throw std::out_of_range("Buffer overflow while reading uint32_t");
    }

    std::uint32_t value = (static_cast<std::uint32_t>(std::to_integer<unsigned char>(buffer[offset])) << 24) |
                          (static_cast<std::uint32_t>(std::to_integer<unsigned char>(buffer[offset + 1])) << 16) |
                          (static_cast<std::uint32_t>(std::to_integer<unsigned char>(buffer[offset + 2])) << 8) |
                          static_cast<std::uint32_t>(std::to_integer<unsigned char>(buffer[offset + 3]));
    offset += 4;
    return value;
}

std::uint64_t distributed::read_uint64(const ByteBuffer &buffer, std::size_t &offset)
{
    if (offset + 8 > buffer.size())
    {
        throw std::out_of_range("Buffer overflow while reading uint64_t");
    }

    std::uint64_t value = (static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset])) << 56) |
                          (static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset + 1])) << 48) |
                          (static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset + 2])) << 40) |
                          (static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset + 3])) << 32) |
                          (static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset + 4])) << 24) |
                          (static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset + 5])) << 16) |
                          (static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset + 6])) << 8) |
                          static_cast<std::uint64_t>(std::to_integer<unsigned char>(buffer[offset + 7]));
    offset += 8;
    return value;
}

float distributed::read_float(const ByteBuffer &buffer, std::size_t &offset)
{
    std::uint32_t as_int = read_uint32(buffer, offset);
    float value;
    std::memcpy(&value, &as_int, sizeof(float));
    return value;
}

double distributed::read_double(const ByteBuffer &buffer, std::size_t &offset)
{
    std::uint64_t as_int = read_uint64(buffer, offset);
    double value;
    std::memcpy(&value, &as_int, sizeof(double));
    return value;
}

void distributed::append_float_vector(ByteBuffer &buffer, const std::vector<float> &values)
{
    for (float value : values)
    {
        append_float(buffer, value);
    }
}

std::vector<float> distributed::read_float_vector(const ByteBuffer &buffer, std::size_t &offset, std::size_t count)
{
    std::vector<float> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        values.push_back(read_float(buffer, offset));
    }
    return values;
}

distributed::ByteBuffer distributed::read_bytes(const ByteBuffer &buffer, std::size_t &offset, std::size_t count)
{
    if (offset + count > buffer.size())
    {
        throw std::out_of_range("Buffer overflow while reading bytes");
    }

    ByteBuffer result(buffer.begin() + offset, buffer.begin() + offset + count);
    offset += count;
    return result;
}
