#include "distributed/protocol.hpp"
#include "distributed/serialization.hpp"

#include "test_support.hpp"

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

        header.magic = distributed::protocol_magic;
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
        CHECK_EQ(actual.magic, expected.magic);
        CHECK_EQ(actual.version, expected.version);
        CHECK_EQ(actual.type, expected.type);
        CHECK_EQ(actual.payload_size, expected.payload_size);
        CHECK_EQ(actual.worker_id, expected.worker_id);
        CHECK_EQ(actual.phase, expected.phase);
        CHECK_EQ(actual.step, expected.step);
        CHECK_EQ(actual.chunk_index, expected.chunk_index);
        CHECK_EQ(actual.element_count, expected.element_count);
    }

} // namespace

// -----------------------------------------------------------------------------
// Message type validation
// -----------------------------------------------------------------------------

TEST_CASE(MessageTypeValidation, AcceptsFirstAndLastMessageTypes)
{
    CHECK_TRUE(distributed::is_valid_message_type(
        static_cast<std::uint16_t>(
            distributed::MessageType::register_worker)));

    CHECK_TRUE(distributed::is_valid_message_type(
        static_cast<std::uint16_t>(
            distributed::MessageType::shutdown)));
}

TEST_CASE(MessageTypeValidation, AcceptsAllValuesInsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::MessageType::register_worker);

    const auto last = static_cast<std::uint32_t>(
        distributed::MessageType::shutdown);

    CHECK_LE(first, last);

    for (std::uint32_t value = first; value <= last; ++value)
    {
        CHECK_TRUE(distributed::is_valid_message_type(
            static_cast<std::uint16_t>(value)))
            << "Unexpected invalid message type: " << value;
    }
}

TEST_CASE(MessageTypeValidation, RejectsValuesOutsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::MessageType::register_worker);

    const auto last = static_cast<std::uint32_t>(
        distributed::MessageType::shutdown);

    if (first > 0)
    {
        CHECK_FALSE(distributed::is_valid_message_type(
            static_cast<std::uint16_t>(first - 1)));
    }

    if (last < std::numeric_limits<std::uint16_t>::max())
    {
        CHECK_FALSE(distributed::is_valid_message_type(
            static_cast<std::uint16_t>(last + 1)));
    }
}

// -----------------------------------------------------------------------------
// Phase validation
// -----------------------------------------------------------------------------

TEST_CASE(PhaseValidation, AcceptsFirstAndLastPhases)
{
    CHECK_TRUE(distributed::is_valid_phase(
        static_cast<std::uint16_t>(
            distributed::Phase::setup)));

    CHECK_TRUE(distributed::is_valid_phase(
        static_cast<std::uint16_t>(
            distributed::Phase::complete)));
}

TEST_CASE(PhaseValidation, AcceptsAllValuesInsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::Phase::setup);

    const auto last = static_cast<std::uint32_t>(
        distributed::Phase::complete);

    CHECK_LE(first, last);

    for (std::uint32_t value = first; value <= last; ++value)
    {
        CHECK_TRUE(distributed::is_valid_phase(
            static_cast<std::uint16_t>(value)))
            << "Unexpected invalid phase: " << value;
    }
}

TEST_CASE(PhaseValidation, RejectsValuesOutsideDeclaredRange)
{
    const auto first = static_cast<std::uint32_t>(
        distributed::Phase::setup);

    const auto last = static_cast<std::uint32_t>(
        distributed::Phase::complete);

    if (first > 0)
    {
        CHECK_FALSE(distributed::is_valid_phase(
            static_cast<std::uint16_t>(first - 1)));
    }

    if (last < std::numeric_limits<std::uint16_t>::max())
    {
        CHECK_FALSE(distributed::is_valid_phase(
            static_cast<std::uint16_t>(last + 1)));
    }
}

// -----------------------------------------------------------------------------
// Header serialization
// -----------------------------------------------------------------------------

TEST_CASE(HeaderSerialization, ProducesExpectedNetworkByteOrder)
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

    CHECK_EQ(actual, expected);
    CHECK_EQ(actual.size(), distributed::message_header_size);
}

TEST_CASE(HeaderSerialization, RoundTripsEveryHeaderField)
{
    const distributed::MessageHeader original =
        make_test_header(128);

    const ByteBuffer encoded =
        distributed::serialize_header(original);

    std::size_t offset = 0;

    const distributed::MessageHeader decoded =
        distributed::deserialize_header(encoded, offset);

    expect_headers_equal(original, decoded);
    CHECK_EQ(offset, distributed::message_header_size);
}

TEST_CASE(HeaderDeserialization, StartsReadingAtProvidedOffset)
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

    CHECK_EQ(
        offset,
        3U + distributed::message_header_size);
}

TEST_CASE(HeaderDeserialization, RejectsTruncatedHeader)
{
    const distributed::MessageHeader header =
        make_test_header();

    ByteBuffer encoded =
        distributed::serialize_header(header);

    CHECK_FALSE(encoded.empty());

    encoded.pop_back();

    std::size_t offset = 0;

    CHECK_ANY_THROW({
        const auto ignored =
            distributed::deserialize_header(encoded, offset);
        (void)ignored;
    });
}

TEST_CASE(HeaderDeserialization, RejectsOffsetPastEndOfBuffer)
{
    const distributed::MessageHeader header =
        make_test_header();

    const ByteBuffer encoded =
        distributed::serialize_header(header);

    std::size_t offset = encoded.size() + 1;

    CHECK_ANY_THROW({
        const auto ignored =
            distributed::deserialize_header(encoded, offset);
        (void)ignored;
    });
}

// -----------------------------------------------------------------------------
// Message serialization
// -----------------------------------------------------------------------------

TEST_CASE(MessageSerialization, PlacesPayloadImmediatelyAfterHeader)
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

    CHECK_EQ(actual, expected);
}

TEST_CASE(MessageSerialization, RoundTripsHeaderAndPayload)
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
    CHECK_EQ(decoded.payload, original.payload);
    CHECK_EQ(offset, encoded.size());
}

TEST_CASE(MessageDeserialization, SupportsEmptyPayload)
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
    CHECK_TRUE(decoded.payload.empty());
    CHECK_EQ(offset, distributed::message_header_size);
}

TEST_CASE(MessageDeserialization, LeavesTrailingBytesUnread)
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
    CHECK_EQ(decoded.payload, original.payload);
    CHECK_EQ(offset, message_size);
    CHECK_LT(offset, buffer.size());
}

TEST_CASE(MessageDeserialization, RejectsTruncatedPayload)
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

    CHECK_FALSE(encoded.empty());

    encoded.pop_back();

    std::size_t offset = 0;

    CHECK_ANY_THROW({
        const auto ignored =
            distributed::deserialize_message(encoded, offset);
        (void)ignored;
    });
}

TEST_CASE(MessageDeserialization, RejectsPayloadWhenNoPayloadBytesExist)
{
    distributed::MessageHeader header =
        make_test_header(8);

    const ByteBuffer encoded =
        distributed::serialize_header(header);

    std::size_t offset = 0;

    CHECK_ANY_THROW({
        const auto ignored =
            distributed::deserialize_message(encoded, offset);
        (void)ignored;
    });
}

TEST_CASE(MessageDeserialization, ReadsConcatenatedMessagesSequentially)
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

    CHECK_EQ(
        decoded_first.payload,
        first.payload);

    expect_headers_equal(
        second.header,
        decoded_second.header);

    CHECK_EQ(
        decoded_second.payload,
        second.payload);

    CHECK_EQ(offset, buffer.size());
}

TEST_CASE(MessageDeserialization, RespectsNonZeroInitialOffset)
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
    CHECK_EQ(decoded.payload, original.payload);
    CHECK_EQ(offset, buffer.size());
}

TEST_CASE(MessageValidation, RejectsInvalidHeadersAndMismatchedPayload)
{
    distributed::Message message;
    message.header.type = distributed::MessageType::ready;
    message.payload = make_bytes({1, 2});
    CHECK_THROW(distributed::serialize_message(message), std::invalid_argument);
    message.header.payload_size = 2;
    CHECK_NO_THROW(distributed::serialize_message(message));
    for (int field = 0; field < 5; ++field)
    {
        auto header = message.header;
        if (field == 0) header.magic = 0;
        if (field == 1) header.version = 999;
        if (field == 2) header.type = static_cast<distributed::MessageType>(0);
        if (field == 3) header.phase = static_cast<distributed::Phase>(999);
        if (field == 4) header.payload_size = distributed::max_payload_size + 1;
        const auto encoded = distributed::serialize_header(header);
        std::size_t offset = 0;
        CHECK_THROW(distributed::deserialize_message(encoded, offset), std::runtime_error);
    }
}
