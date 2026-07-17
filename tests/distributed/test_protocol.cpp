#include "distributed/protocol.hpp"
#include "distributed/serialization.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace
{

    using distributed::ByteBuffer;
    using Byte = ByteBuffer::value_type;

    ByteBuffer make_bytes(std::initializer_list<std::uint8_t> values)
    {
        ByteBuffer buffer;
        buffer.reserve(values.size());

        for (const std::uint8_t value : values)
        {
            buffer.push_back(static_cast<Byte>(value));
        }

        return buffer;
    }

    distributed::MessageHeader make_test_header(
        std::uint32_t payload_size = 0)
    {
        distributed::MessageHeader header{};

        header.magic = 0x52415231;
        header.version = 1;
        header.type = distributed::MessageType::register_worker;
        header.payload_size = payload_size;
        header.worker_id = 7;
        header.phase = distributed::Phase::setup;
        header.step = 3;
        header.chunk_index = 2;
        header.element_count = 16;

        return header;
    }

    void expect_headers_equal(
        const distributed::MessageHeader &expected,
        const distributed::MessageHeader &actual)
    {
        EXPECT_EQ(actual.magic, expected.magic);
        EXPECT_EQ(actual.version, expected.version);
        EXPECT_EQ(actual.type, expected.type);
        EXPECT_EQ(actual.payload_size, expected.payload_size);
        EXPECT_EQ(actual.worker_id, expected.worker_id);
        EXPECT_EQ(actual.phase, expected.phase);
        EXPECT_EQ(actual.step, expected.step);
        EXPECT_EQ(actual.chunk_index, expected.chunk_index);
        EXPECT_EQ(actual.element_count, expected.element_count);
    }

} // namespace

// -----------------------------------------------------------------------------
// Message type validation
// -----------------------------------------------------------------------------

TEST(MessageTypeValidation, AcceptsFirstAndLastMessageTypes)
{
    EXPECT_TRUE(distributed::is_valid_message_type(
        static_cast<std::uint16_t>(
            distributed::MessageType::register_worker)));

    EXPECT_TRUE(distributed::is_valid_message_type(
        static_cast<std::uint16_t>(
            distributed::MessageType::shutdown)));
}

TEST(MessageTypeValidation, AcceptsAllValuesInsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::MessageType::register_worker);

    const auto last = static_cast<std::uint32_t>(
        distributed::MessageType::shutdown);

    ASSERT_LE(first, last);

    for (std::uint32_t value = first; value <= last; ++value)
    {
        EXPECT_TRUE(distributed::is_valid_message_type(
            static_cast<std::uint16_t>(value)))
            << "Unexpected invalid message type: " << value;
    }
}

TEST(MessageTypeValidation, RejectsValuesOutsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::MessageType::register_worker);

    const auto last = static_cast<std::uint32_t>(
        distributed::MessageType::shutdown);

    if (first > 0)
    {
        EXPECT_FALSE(distributed::is_valid_message_type(
            static_cast<std::uint16_t>(first - 1)));
    }

    if (last < std::numeric_limits<std::uint16_t>::max())
    {
        EXPECT_FALSE(distributed::is_valid_message_type(
            static_cast<std::uint16_t>(last + 1)));
    }
}

// -----------------------------------------------------------------------------
// Phase validation
// -----------------------------------------------------------------------------

TEST(PhaseValidation, AcceptsFirstAndLastPhases)
{
    EXPECT_TRUE(distributed::is_valid_phase(
        static_cast<std::uint16_t>(
            distributed::Phase::setup)));

    EXPECT_TRUE(distributed::is_valid_phase(
        static_cast<std::uint16_t>(
            distributed::Phase::complete)));
}

TEST(PhaseValidation, AcceptsAllValuesInsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::Phase::setup);

    const auto last = static_cast<std::uint32_t>(
        distributed::Phase::complete);

    ASSERT_LE(first, last);

    for (std::uint32_t value = first; value <= last; ++value)
    {
        EXPECT_TRUE(distributed::is_valid_phase(
            static_cast<std::uint16_t>(value)))
            << "Unexpected invalid phase: " << value;
    }
}

TEST(PhaseValidation, RejectsValuesOutsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::Phase::setup);

    const auto last = static_cast<std::uint32_t>(
        distributed::Phase::complete);

    if (first > 0)
    {
        EXPECT_FALSE(distributed::is_valid_phase(
            static_cast<std::uint16_t>(first - 1)));
    }

    if (last < std::numeric_limits<std::uint16_t>::max())
    {
        EXPECT_FALSE(distributed::is_valid_phase(
            static_cast<std::uint16_t>(last + 1)));
    }
}

// -----------------------------------------------------------------------------
// Header serialization
// -----------------------------------------------------------------------------

TEST(HeaderSerialization, ProducesExpectedNetworkByteOrder)
{
    distributed::MessageHeader header{};

    header.magic = 0x01020304;
    header.version = 0x0506;
    header.type =
        static_cast<distributed::MessageType>(0x0708);
    header.payload_size = 0x090A0B0C;
    header.worker_id = 0x0D0E0F10;
    header.phase =
        static_cast<distributed::Phase>(0x1112);
    header.step = 0x13141516;
    header.chunk_index = 0x1718191A;
    header.element_count = 0x1B1C1D1E;

    const ByteBuffer expected = make_bytes({
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x07,
        0x08,
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x0E,
        0x0F,
        0x10,
        0x11,
        0x12,
        0x13,
        0x14,
        0x15,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1A,
        0x1B,
        0x1C,
        0x1D,
        0x1E,
    });

    const ByteBuffer actual =
        distributed::serialize_header(header);

    EXPECT_EQ(actual, expected);
    EXPECT_EQ(actual.size(), distributed::message_header_size);
}

TEST(HeaderSerialization, RoundTripsEveryHeaderField)
{
    const distributed::MessageHeader original =
        make_test_header(128);

    const ByteBuffer encoded =
        distributed::serialize_header(original);

    std::size_t offset = 0;

    const distributed::MessageHeader decoded =
        distributed::deserialize_header(encoded, offset);

    expect_headers_equal(original, decoded);
    EXPECT_EQ(offset, distributed::message_header_size);
}

TEST(HeaderDeserialization, StartsReadingAtProvidedOffset)
{
    const distributed::MessageHeader original =
        make_test_header(32);

    const ByteBuffer encoded =
        distributed::serialize_header(original);

    ByteBuffer buffer = make_bytes({
        0xAA,
        0xBB,
        0xCC,
    });

    buffer.insert(
        buffer.end(),
        encoded.begin(),
        encoded.end());

    buffer.push_back(static_cast<Byte>(0xDD));

    std::size_t offset = 3;

    const distributed::MessageHeader decoded =
        distributed::deserialize_header(buffer, offset);

    expect_headers_equal(original, decoded);

    EXPECT_EQ(
        offset,
        3U + distributed::message_header_size);
}

TEST(HeaderDeserialization, RejectsTruncatedHeader)
{
    const distributed::MessageHeader header =
        make_test_header();

    ByteBuffer encoded =
        distributed::serialize_header(header);

    ASSERT_FALSE(encoded.empty());

    encoded.pop_back();

    std::size_t offset = 0;

    EXPECT_ANY_THROW({
        const auto ignored =
            distributed::deserialize_header(encoded, offset);
        (void)ignored;
    });
}

TEST(HeaderDeserialization, RejectsOffsetPastEndOfBuffer)
{
    const distributed::MessageHeader header =
        make_test_header();

    const ByteBuffer encoded =
        distributed::serialize_header(header);

    std::size_t offset = encoded.size() + 1;

    EXPECT_ANY_THROW({
        const auto ignored =
            distributed::deserialize_header(encoded, offset);
        (void)ignored;
    });
}

// -----------------------------------------------------------------------------
// Message serialization
// -----------------------------------------------------------------------------

TEST(MessageSerialization, PlacesPayloadImmediatelyAfterHeader)
{
    distributed::Message message{};

    message.payload = make_bytes({
        0xDE,
        0xAD,
        0xBE,
        0xEF,
    });

    message.header = make_test_header(
        static_cast<std::uint32_t>(message.payload.size()));

    ByteBuffer expected =
        distributed::serialize_header(message.header);

    expected.insert(
        expected.end(),
        message.payload.begin(),
        message.payload.end());

    const ByteBuffer actual =
        distributed::serialize_message(message);

    EXPECT_EQ(actual, expected);
}

TEST(MessageSerialization, RoundTripsHeaderAndPayload)
{
    distributed::Message original{};

    original.payload = make_bytes({
        0x10,
        0x20,
        0x30,
        0x40,
        0x50,
    });

    original.header = make_test_header(
        static_cast<std::uint32_t>(original.payload.size()));

    const ByteBuffer encoded =
        distributed::serialize_message(original);

    std::size_t offset = 0;

    const distributed::Message decoded =
        distributed::deserialize_message(encoded, offset);

    expect_headers_equal(original.header, decoded.header);
    EXPECT_EQ(decoded.payload, original.payload);
    EXPECT_EQ(offset, encoded.size());
}

TEST(MessageDeserialization, SupportsEmptyPayload)
{
    distributed::Message original{};

    original.header = make_test_header(0);
    original.payload.clear();

    const ByteBuffer encoded =
        distributed::serialize_message(original);

    std::size_t offset = 0;

    const distributed::Message decoded =
        distributed::deserialize_message(encoded, offset);

    expect_headers_equal(original.header, decoded.header);
    EXPECT_TRUE(decoded.payload.empty());
    EXPECT_EQ(offset, distributed::message_header_size);
}

TEST(MessageDeserialization, LeavesTrailingBytesUnread)
{
    distributed::Message original{};

    original.payload = make_bytes({
        0x01,
        0x02,
        0x03,
    });

    original.header = make_test_header(
        static_cast<std::uint32_t>(original.payload.size()));

    ByteBuffer buffer =
        distributed::serialize_message(original);

    const std::size_t message_size = buffer.size();

    buffer.push_back(static_cast<Byte>(0xFE));
    buffer.push_back(static_cast<Byte>(0xED));

    std::size_t offset = 0;

    const distributed::Message decoded =
        distributed::deserialize_message(buffer, offset);

    expect_headers_equal(original.header, decoded.header);
    EXPECT_EQ(decoded.payload, original.payload);
    EXPECT_EQ(offset, message_size);
    EXPECT_LT(offset, buffer.size());
}

TEST(MessageDeserialization, RejectsTruncatedPayload)
{
    distributed::Message message{};

    message.payload = make_bytes({
        0x01,
        0x02,
        0x03,
        0x04,
    });

    message.header = make_test_header(
        static_cast<std::uint32_t>(message.payload.size()));

    ByteBuffer encoded =
        distributed::serialize_message(message);

    ASSERT_FALSE(encoded.empty());

    encoded.pop_back();

    std::size_t offset = 0;

    EXPECT_ANY_THROW({
        const auto ignored =
            distributed::deserialize_message(encoded, offset);
        (void)ignored;
    });
}

TEST(MessageDeserialization, RejectsPayloadWhenNoPayloadBytesExist)
{
    distributed::MessageHeader header =
        make_test_header(8);

    const ByteBuffer encoded =
        distributed::serialize_header(header);

    std::size_t offset = 0;

    EXPECT_ANY_THROW({
        const auto ignored =
            distributed::deserialize_message(encoded, offset);
        (void)ignored;
    });
}

TEST(MessageDeserialization, ReadsConcatenatedMessagesSequentially)
{
    distributed::Message first{};

    first.payload = make_bytes({
        0x01,
        0x02,
        0x03,
    });

    first.header = make_test_header(
        static_cast<std::uint32_t>(first.payload.size()));

    first.header.worker_id = 1;
    first.header.step = 10;

    distributed::Message second{};

    second.payload = make_bytes({
        0xAA,
        0xBB,
    });

    second.header = make_test_header(
        static_cast<std::uint32_t>(second.payload.size()));

    second.header.worker_id = 2;
    second.header.step = 11;
    second.header.type =
        distributed::MessageType::shutdown;
    second.header.phase =
        distributed::Phase::complete;

    ByteBuffer buffer =
        distributed::serialize_message(first);

    const ByteBuffer second_buffer =
        distributed::serialize_message(second);

    buffer.insert(
        buffer.end(),
        second_buffer.begin(),
        second_buffer.end());

    std::size_t offset = 0;

    const distributed::Message decoded_first =
        distributed::deserialize_message(buffer, offset);

    const distributed::Message decoded_second =
        distributed::deserialize_message(buffer, offset);

    expect_headers_equal(
        first.header,
        decoded_first.header);

    EXPECT_EQ(
        decoded_first.payload,
        first.payload);

    expect_headers_equal(
        second.header,
        decoded_second.header);

    EXPECT_EQ(
        decoded_second.payload,
        second.payload);

    EXPECT_EQ(offset, buffer.size());
}

TEST(MessageDeserialization, RespectsNonZeroInitialOffset)
{
    distributed::Message original{};

    original.payload = make_bytes({
        0x12,
        0x34,
        0x56,
    });

    original.header = make_test_header(
        static_cast<std::uint32_t>(original.payload.size()));

    const ByteBuffer encoded =
        distributed::serialize_message(original);

    ByteBuffer buffer = make_bytes({
        0x99,
        0x88,
    });

    buffer.insert(
        buffer.end(),
        encoded.begin(),
        encoded.end());

    std::size_t offset = 2;

    const distributed::Message decoded =
        distributed::deserialize_message(buffer, offset);

    expect_headers_equal(original.header, decoded.header);
    EXPECT_EQ(decoded.payload, original.payload);
    EXPECT_EQ(offset, buffer.size());
}
