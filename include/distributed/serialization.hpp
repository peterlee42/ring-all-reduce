#pragma once

#include <vector>
#include <cstdint>

namespace distributed
{
    // type alias for a buffer of bytes
    using ByteBuffer = std::vector<std::byte>;

    void append_uint16(ByteBuffer &buffer, std::uint16_t value);
    void append_uint32(ByteBuffer &buffer, std::uint32_t value);
    void append_uint64(ByteBuffer &buffer, std::uint64_t value);
    void append_float(ByteBuffer &buffer, float value);
    void append_double(ByteBuffer &buffer, double value);

    std::uint16_t read_uint16(const ByteBuffer &buffer, std::size_t &offset);
    std::uint32_t read_uint32(const ByteBuffer &buffer, std::size_t &offset);
    std::uint64_t read_uint64(const ByteBuffer &buffer, std::size_t &offset);
    float read_float(const ByteBuffer &buffer, std::size_t &offset);
    double read_double(const ByteBuffer &buffer, std::size_t &offset);

    void append_float_vector(ByteBuffer &buffer, const std::vector<float> &values);
    std::vector<float> read_float_vector(const ByteBuffer &buffer, std::size_t &offset, std::size_t count);
    ByteBuffer read_bytes(const ByteBuffer &buffer, std::size_t &offset, std::size_t count);
}
