#include "distributed/serialization.hpp"

#include "test_support.hpp"

#include <cstddef>
#include <cstdint>

namespace
{
    TEST_CASE(SerializationTest, AppendsAndReadsUint16)
    {
        distributed::ByteBuffer buffer;

        distributed::append_uint16(buffer, 0x1234U);

        CHECK_EQ(buffer.size(), 2U);

        CHECK_EQ(std::to_integer<unsigned int>(buffer[0]), 0x12U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[1]), 0x34U);

        std::size_t offset = 0;

        const std::uint16_t decoded =
            distributed::read_uint16(buffer, offset);

        CHECK_EQ(decoded, 0x1234U);
        CHECK_EQ(offset, 2U);
    }

    TEST_CASE(SerializationTest, AppendsAndReadsUint32)
    {
        distributed::ByteBuffer buffer;

        distributed::append_uint32(buffer, 0x12345678U);

        CHECK_EQ(buffer.size(), 4U);

        CHECK_EQ(std::to_integer<unsigned int>(buffer[0]), 0x12U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[1]), 0x34U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[2]), 0x56U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[3]), 0x78U);

        std::size_t offset = 0;

        const std::uint32_t decoded =
            distributed::read_uint32(buffer, offset);

        CHECK_EQ(decoded, 0x12345678U);
        CHECK_EQ(offset, 4U);
    }

    TEST_CASE(SerializationTest, AppendsAndReadsUint64)
    {
        distributed::ByteBuffer buffer;

        distributed::append_uint64(buffer, 0x123456789ABCDEF0ULL);

        CHECK_EQ(buffer.size(), 8U);

        CHECK_EQ(std::to_integer<unsigned int>(buffer[0]), 0x12U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[1]), 0x34U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[2]), 0x56U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[3]), 0x78U);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[4]), 0x9AU);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[5]), 0xBCU);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[6]), 0xDEU);
        CHECK_EQ(std::to_integer<unsigned int>(buffer[7]), 0xF0U);

        std::size_t offset = 0;

        const std::uint64_t decoded =
            distributed::read_uint64(buffer, offset);

        CHECK_EQ(decoded, 0x123456789ABCDEF0ULL);
        CHECK_EQ(offset, 8U);
    }

    TEST_CASE(SerializationTest, AppendsAndReadsFloat)
    {
        distributed::ByteBuffer buffer;

        distributed::append_float(buffer, 3.14f);

        CHECK_EQ(buffer.size(), 4U);

        std::size_t offset = 0;

        const float decoded =
            distributed::read_float(buffer, offset);

        CHECK_FLOAT_EQ(decoded, 3.14f);
        CHECK_EQ(offset, 4U);
    }

    TEST_CASE(SerializationTest, AppendsAndReadsDouble)
    {
        distributed::ByteBuffer buffer;

        distributed::append_double(buffer, 3.141592653589793);

        CHECK_EQ(buffer.size(), 8U);

        std::size_t offset = 0;

        const double decoded =
            distributed::read_double(buffer, offset);

        CHECK_DOUBLE_EQ(decoded, 3.141592653589793);
        CHECK_EQ(offset, 8U);
    }

    TEST_CASE(SerializationTest, AppendsAndReadsFloatVector)
    {
        distributed::ByteBuffer buffer;

        std::vector<float> original = {1.0f, 2.0f, 3.0f};
        distributed::append_float_vector(buffer, original);

        CHECK_EQ(buffer.size(), 12U);

        std::size_t offset = 0;

        const std::vector<float> decoded =
            distributed::read_float_vector(buffer, offset, original.size());

        CHECK_EQ(decoded.size(), original.size());
        for (std::size_t i = 0; i < original.size(); ++i)
        {
            CHECK_FLOAT_EQ(decoded[i], original[i]);
        }
        CHECK_EQ(offset, 12U);
    }

    TEST_CASE(SerializationTest, ReadsBytes)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
            std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};

        std::size_t offset = 2;
        const std::size_t count = 4;

        const distributed::ByteBuffer bytes =
            distributed::read_bytes(buffer, offset, count);

        CHECK_EQ(bytes.size(), count);
        CHECK_EQ(std::to_integer<unsigned int>(bytes[0]), 0x03U);
        CHECK_EQ(std::to_integer<unsigned int>(bytes[1]), 0x04U);
        CHECK_EQ(std::to_integer<unsigned int>(bytes[2]), 0x05U);
        CHECK_EQ(std::to_integer<unsigned int>(bytes[3]), 0x06U);
    }

    TEST_CASE(SerializationTest, ReadsBytesWithOffset)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
            std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};

        std::size_t offset = 4;
        const std::size_t count = 2;

        const distributed::ByteBuffer bytes =
            distributed::read_bytes(buffer, offset, count);

        CHECK_EQ(bytes.size(), count);
        CHECK_EQ(std::to_integer<unsigned int>(bytes[0]), 0x05U);
        CHECK_EQ(std::to_integer<unsigned int>(bytes[1]), 0x06U);
    }

    TEST_CASE(SerializationTest, ReadsBytesWithCountExceedingBuffer)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};

        std::size_t offset = 2;
        const std::size_t count = 4;

        CHECK_THROW(distributed::read_bytes(buffer, offset, count), std::out_of_range);
    }

    TEST_CASE(SerializationTest, ReadsBytesWithOffsetExceedingBuffer)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};

        std::size_t offset = 5;
        const std::size_t count = 2;

        CHECK_THROW(distributed::read_bytes(buffer, offset, count), std::out_of_range);
    }

    TEST_CASE(SerializationTest, ThrowsForTruncatedReadUint16)
    {
        distributed::ByteBuffer buffer = {std::byte{0x01}};

        std::size_t offset = 0;

        CHECK_THROW(distributed::read_uint16(buffer, offset), std::out_of_range);
    }
}



TEST_CASE(SerializationBounds, RejectsOverflowingOffsetsAndCountsBeforeAllocation)
{
    const distributed::ByteBuffer buffer{std::byte{0}};
    std::size_t offset = static_cast<std::size_t>(-1);
    CHECK_THROW(distributed::read_uint16(buffer, offset), std::out_of_range);
    CHECK_THROW(distributed::read_uint32(buffer, offset), std::out_of_range);
    CHECK_THROW(distributed::read_uint64(buffer, offset), std::out_of_range);
    CHECK_THROW(distributed::read_bytes(buffer, offset, 2), std::out_of_range);
    offset = 1;
    CHECK_THROW(distributed::read_bytes(buffer, offset, static_cast<std::size_t>(-1)), std::out_of_range);
    CHECK_THROW(distributed::read_float_vector(buffer, offset, static_cast<std::size_t>(-1)), std::out_of_range);
}
