#pragma once

#include "distributed/serialization.hpp"

#include <cstddef>
#include <cstdint>

namespace distributed
{
    inline constexpr std::uint32_t protocol_magic = 0x52414C52; // "RALR" in ASCII
    inline constexpr std::uint16_t protocol_version = 1U;
    inline constexpr std::size_t message_header_size = 30U; // 30 bytes

    inline constexpr std::uint32_t max_payload_size = 16U * 1024U * 1024U; // 16 MB

    enum class MessageType : std::uint16_t
    {
        register_worker = 1,
        register_ack = 2,
        ring_config = 3,
        ready = 4,
        start = 5,
        reduce_chunk = 6,
        gather_chunk = 7,
        round_done = 8,
        training_done = 9,
        error = 10,
        shutdown = 11
    };

    enum class Phase : std::uint16_t
    {
        setup = 0,
        reduce_scatter = 1,
        all_gather = 2,
        complete = 3,
    };

    struct MessageHeader
    {
        std::uint32_t magic = protocol_magic;
        std::uint16_t version = protocol_version;
        MessageType type = MessageType::error;
        std::uint32_t payload_size = 0;
        std::uint32_t worker_id = 0;
        Phase phase = Phase::setup;
        std::uint32_t step = 0;
        std::uint32_t chunk_index = 0;
        std::uint32_t element_count = 0;
    };

    struct Message
    {
        MessageHeader header;
        ByteBuffer payload;
    };

    bool is_valid_message_type(std::uint16_t value) noexcept;
    bool is_valid_phase(std::uint16_t value) noexcept;

    ByteBuffer serialize_header(const MessageHeader &header);
    MessageHeader deserialize_header(const ByteBuffer &buffer, std::size_t &offset);

    ByteBuffer serialize_message(const Message &message);
    Message deserialize_message(const ByteBuffer &buffer, std::size_t &offset);
}
