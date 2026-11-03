#pragma once

#include "distributed/protocol.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace distributed
{
    struct RingOptions
    {
        std::size_t rank = 0;
        std::size_t workers = 1;
        std::uint16_t base_port = 5000;
        // One IPv4 address per rank; empty means all workers use loopback.
        std::vector<std::string> hosts;
        std::chrono::milliseconds timeout{10000};
        // Optional connected stream sockets, owned by Ring after construction.
        // The local launcher uses inherited socketpairs to avoid TCP ports.
        int incoming_fd = -1;
        int outgoing_fd = -1;
    };

    // A synchronous collective over separate processes. Every rank must call
    // collectives in the same order. A failed collective invalidates the ring.
    class Ring
    {
    public:
        explicit Ring(const RingOptions &options);
        ~Ring();
        Ring(const Ring &) = delete;
        Ring &operator=(const Ring &) = delete;

        void verify_configuration(const ByteBuffer &configuration);
        void all_reduce_sum(std::vector<double> &values);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
