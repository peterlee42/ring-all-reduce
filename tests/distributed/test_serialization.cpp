#include "distributed/serialization.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace
{
    TEST(SerializationTest, AppendsAndReadsUint16)
    {
        distributed::ByteBuffer buffer;

        distributed::append_uint16(buffer, 0x1234U);

        ASSERT_EQ(buffer.size(), 2U);

        EXPECT_EQ(std::to_integer<unsigned int>(buffer[0]), 0x12U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[1]), 0x34U);

        std::size_t offset = 0;

        const std::uint16_t decoded =
            distributed::read_uint16(buffer, offset);

        EXPECT_EQ(decoded, 0x1234U);
        EXPECT_EQ(offset, 2U);
    }

    TEST(SerializationTest, AppendsAndReadsUint32)
    {
        distributed::ByteBuffer buffer;

        distributed::append_uint32(buffer, 0x12345678U);

        ASSERT_EQ(buffer.size(), 4U);

        EXPECT_EQ(std::to_integer<unsigned int>(buffer[0]), 0x12U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[1]), 0x34U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[2]), 0x56U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[3]), 0x78U);

        std::size_t offset = 0;

        const std::uint32_t decoded =
            distributed::read_uint32(buffer, offset);

        EXPECT_EQ(decoded, 0x12345678U);
        EXPECT_EQ(offset, 4U);
    }

    TEST(SerializationTest, AppendsAndReadsUint64)
    {
        distributed::ByteBuffer buffer;

        distributed::append_uint64(buffer, 0x123456789ABCDEF0ULL);

        ASSERT_EQ(buffer.size(), 8U);

        EXPECT_EQ(std::to_integer<unsigned int>(buffer[0]), 0x12U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[1]), 0x34U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[2]), 0x56U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[3]), 0x78U);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[4]), 0x9AU);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[5]), 0xBCU);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[6]), 0xDEU);
        EXPECT_EQ(std::to_integer<unsigned int>(buffer[7]), 0xF0U);

        std::size_t offset = 0;

        const std::uint64_t decoded =
            distributed::read_uint64(buffer, offset);

        EXPECT_EQ(decoded, 0x123456789ABCDEF0ULL);
        EXPECT_EQ(offset, 8U);
    }

    TEST(SerializationTest, AppendsAndReadsFloat)
    {
        distributed::ByteBuffer buffer;

        distributed::append_float(buffer, 3.14f);

        ASSERT_EQ(buffer.size(), 4U);

        std::size_t offset = 0;

        const float decoded =
            distributed::read_float(buffer, offset);

        EXPECT_FLOAT_EQ(decoded, 3.14f);
        EXPECT_EQ(offset, 4U);
    }

    TEST(SerializationTest, AppendsAndReadsDouble)
    {
        distributed::ByteBuffer buffer;

        distributed::append_double(buffer, 3.141592653589793);

        ASSERT_EQ(buffer.size(), 8U);

        std::size_t offset = 0;

        const double decoded =
            distributed::read_double(buffer, offset);

        EXPECT_DOUBLE_EQ(decoded, 3.141592653589793);
        EXPECT_EQ(offset, 8U);
    }

    TEST(SerializationTest, AppendsAndReadsFloatVector)
    {
        distributed::ByteBuffer buffer;

        std::vector<float> original = {1.0f, 2.0f, 3.0f};
        distributed::append_float_vector(buffer, original);

        ASSERT_EQ(buffer.size(), 12U);

        std::size_t offset = 0;

        const std::vector<float> decoded =
            distributed::read_float_vector(buffer, offset, original.size());

        EXPECT_EQ(decoded.size(), original.size());
        for (std::size_t i = 0; i < original.size(); ++i)
        {
            EXPECT_FLOAT_EQ(decoded[i], original[i]);
        }
        EXPECT_EQ(offset, 12U);
    }

    TEST(SerializationTest, ReadsBytes)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
            std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};

        std::size_t offset = 2;
        const std::size_t count = 4;

        const distributed::ByteBuffer bytes =
            distributed::read_bytes(buffer, offset, count);

        ASSERT_EQ(bytes.size(), count);
        EXPECT_EQ(std::to_integer<unsigned int>(bytes[0]), 0x03U);
        EXPECT_EQ(std::to_integer<unsigned int>(bytes[1]), 0x04U);
        EXPECT_EQ(std::to_integer<unsigned int>(bytes[2]), 0x05U);
        EXPECT_EQ(std::to_integer<unsigned int>(bytes[3]), 0x06U);
    }

    TEST(SerializationTest, ReadsBytesWithOffset)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04},
            std::byte{0x05}, std::byte{0x06}, std::byte{0x07}, std::byte{0x08}};

        std::size_t offset = 4;
        const std::size_t count = 2;

        const distributed::ByteBuffer bytes =
            distributed::read_bytes(buffer, offset, count);

        ASSERT_EQ(bytes.size(), count);
        EXPECT_EQ(std::to_integer<unsigned int>(bytes[0]), 0x05U);
        EXPECT_EQ(std::to_integer<unsigned int>(bytes[1]), 0x06U);
    }

    TEST(SerializationTest, ReadsBytesWithCountExceedingBuffer)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};

        std::size_t offset = 2;
        const std::size_t count = 4;

        EXPECT_THROW(distributed::read_bytes(buffer, offset, count), std::out_of_range);
    }

    TEST(SerializationTest, ReadsBytesWithOffsetExceedingBuffer)
    {
        distributed::ByteBuffer buffer = {
            std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};

        std::size_t offset = 5;
        const std::size_t count = 2;

        EXPECT_THROW(distributed::read_bytes(buffer, offset, count), std::out_of_range);
    }

    TEST(SerializationTest, ThrowsForTruncatedReadUint16)
    {
        distributed::ByteBuffer buffer = {std::byte{0x01}};

        std::size_t offset = 0;

        EXPECT_THROW(distributed::read_uint16(buffer, offset), std::out_of_range);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
