#include "distributed/protocol.hpp"
#include "distributed/serialization.hpp"

bool distributed::is_valid_message_type(std::uint16_t value) noexcept
{
    return (value >= static_cast<std::uint16_t>(MessageType::register_worker) &&
            value <= static_cast<std::uint16_t>(MessageType::shutdown));
}

bool distributed::is_valid_phase(std::uint16_t value) noexcept
{
    return (value >= static_cast<std::uint16_t>(distributed::Phase::setup) &&
            value <= static_cast<std::uint16_t>(distributed::Phase::complete));
}

distributed::ByteBuffer distributed::serialize_header(const distributed::MessageHeader &header)
{
    distributed::ByteBuffer buffer;
    buffer.reserve(distributed::message_header_size);

    append_uint32(buffer, header.magic);
    append_uint16(buffer, header.version);
    append_uint16(buffer, static_cast<std::uint16_t>(header.type));
    append_uint32(buffer, header.payload_size);
    append_uint32(buffer, header.worker_id);
    append_uint16(buffer, static_cast<std::uint16_t>(header.phase));
    append_uint32(buffer, header.step);
    append_uint32(buffer, header.chunk_index);
    append_uint32(buffer, header.element_count);

    return buffer;
}

distributed::MessageHeader distributed::deserialize_header(const distributed::ByteBuffer &buffer, std::size_t &offset)
{
    distributed::MessageHeader header;

    header.magic = read_uint32(buffer, offset);
    header.version = read_uint16(buffer, offset);
    header.type = static_cast<MessageType>(read_uint16(buffer, offset));
    header.payload_size = read_uint32(buffer, offset);
    header.worker_id = read_uint32(buffer, offset);
    header.phase = static_cast<Phase>(read_uint16(buffer, offset));
    header.step = read_uint32(buffer, offset);
    header.chunk_index = read_uint32(buffer, offset);
    header.element_count = read_uint32(buffer, offset);

    return header;
}

distributed::ByteBuffer distributed::serialize_message(const Message &message)
{
    distributed::ByteBuffer buffer = distributed::serialize_header(message.header);

    buffer.insert(buffer.end(), message.payload.begin(), message.payload.end());

    return buffer;
}

distributed::Message distributed::deserialize_message(const distributed::ByteBuffer &buffer, std::size_t &offset)
{
    distributed::Message message;
    message.header = distributed::deserialize_header(buffer, offset);

    if (message.header.payload_size > 0)
    {
        message.payload = read_bytes(buffer, offset, message.header.payload_size);
    }

    return message;
}
