#include "distributed/ring.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace
{
    using Clock = std::chrono::steady_clock;
    using distributed::ByteBuffer;
    using distributed::Message;

    [[noreturn]] void socket_error(const char *operation)
    {
        throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
    }

    class Socket
    {
    public:
        explicit Socket(int fd = -1) : fd_(fd) {}
        ~Socket() { if (fd_ >= 0) ::close(fd_); }
        Socket(const Socket &) = delete;
        Socket &operator=(const Socket &) = delete;
        Socket(Socket &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
        Socket &operator=(Socket &&other) noexcept
        {
            if (this != &other)
            {
                if (fd_ >= 0) ::close(fd_);
                fd_ = std::exchange(other.fd_, -1);
            }
            return *this;
        }
        int get() const { return fd_; }
    private:
        int fd_;
    };

    void configure(int fd, bool tcp = true)
    {
        const int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
            socket_error("fcntl");
        const int enabled = 1;
        if (tcp && ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) < 0)
            socket_error("TCP_NODELAY");
#ifdef SO_NOSIGPIPE
        if (::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) < 0)
            socket_error("SO_NOSIGPIPE");
#endif
    }

    Socket make_socket()
    {
        Socket socket(::socket(AF_INET, SOCK_STREAM, 0));
        if (socket.get() < 0) socket_error("socket");
        configure(socket.get());
        return socket;
    }

    int remaining_ms(Clock::time_point deadline)
    {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (left <= 0) throw std::runtime_error("Ring operation timed out (missing or stalled worker)");
        return static_cast<int>(std::min<std::int64_t>(left + 1, std::numeric_limits<int>::max()));
    }

    void wait_for(int fd, short events, Clock::time_point deadline)
    {
        pollfd descriptor{fd, events, 0};
        for (;;)
        {
            const int result = ::poll(&descriptor, 1, remaining_ms(deadline));
            if (result > 0) return;
            if (result < 0 && errno != EINTR) socket_error("poll");
        }
    }

    sockaddr_in address(const std::string &host, std::uint16_t port)
    {
        sockaddr_in result{};
        result.sin_family = AF_INET;
        result.sin_port = htons(port);
        if (::inet_pton(AF_INET, host.c_str(), &result.sin_addr) != 1)
            throw std::invalid_argument("Hosts must be numeric IPv4 addresses: " + host);
        return result;
    }

    Socket listen_on(std::uint16_t port)
    {
        Socket listener = make_socket();
        const int enabled = 1;
        if (::setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) < 0)
            socket_error("SO_REUSEADDR");
        const auto endpoint = address("0.0.0.0", port);
        if (::bind(listener.get(), reinterpret_cast<const sockaddr *>(&endpoint), sizeof(endpoint)) < 0)
            socket_error("bind");
        if (::listen(listener.get(), 1) < 0) socket_error("listen");
        return listener;
    }

    Socket connect_to(const std::string &host, std::uint16_t port, Clock::time_point deadline)
    {
        const auto endpoint = address(host, port);
        for (;;)
        {
            (void)remaining_ms(deadline);
            Socket socket = make_socket();
            int error = 0;
            if (::connect(socket.get(), reinterpret_cast<const sockaddr *>(&endpoint), sizeof(endpoint)) < 0)
            {
                error = errno;
                if (error == EINPROGRESS || error == EINTR)
                {
                    wait_for(socket.get(), POLLOUT, deadline);
                    socklen_t size = sizeof(error);
                    if (::getsockopt(socket.get(), SOL_SOCKET, SO_ERROR, &error, &size) < 0)
                        socket_error("getsockopt");
                }
            }
            if (error == 0) return socket;
            if (error != ECONNREFUSED && error != ETIMEDOUT && error != EHOSTUNREACH && error != ENETUNREACH)
                throw std::runtime_error(std::string("connect: ") + std::strerror(error));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    Socket accept_from(int listener, Clock::time_point deadline)
    {
        for (;;)
        {
            wait_for(listener, POLLIN, deadline);
            Socket socket(::accept(listener, nullptr, nullptr));
            if (socket.get() >= 0)
            {
                configure(socket.get());
                return socket;
            }
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) socket_error("accept");
        }
    }

    // Advance both directions with poll. Sending a complete chunk before reading
    // can deadlock every rank when chunks exceed TCP socket buffer capacity.
    Message exchange(int outgoing, int incoming, const Message &message, Clock::time_point deadline)
    {
        const ByteBuffer send_buffer = distributed::serialize_message(message);
        ByteBuffer receive_buffer(distributed::message_header_size);
        std::size_t sent = 0;
        std::size_t received = 0;
        bool header_read = false;
        while (sent < send_buffer.size() || received < receive_buffer.size() || !header_read)
        {
            pollfd descriptors[2] = {
                {sent < send_buffer.size() ? outgoing : -1, POLLOUT, 0},
                {received < receive_buffer.size() ? incoming : -1, POLLIN, 0}};
            const int result = ::poll(descriptors, 2, remaining_ms(deadline));
            if (result < 0)
            {
                if (errno == EINTR) continue;
                socket_error("poll");
            }
            if (descriptors[0].revents & POLLOUT)
            {
#ifdef MSG_NOSIGNAL
                constexpr int flags = MSG_NOSIGNAL;
#else
                constexpr int flags = 0;
#endif
                const auto count = ::send(outgoing, send_buffer.data() + sent, send_buffer.size() - sent, flags);
                if (count > 0) sent += static_cast<std::size_t>(count);
                else if (count < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
                    socket_error("send");
            }
            if (descriptors[1].revents & (POLLIN | POLLHUP))
            {
                const auto count = ::recv(incoming, receive_buffer.data() + received, receive_buffer.size() - received, 0);
                if (count == 0) throw std::runtime_error("Ring peer disconnected");
                if (count > 0) received += static_cast<std::size_t>(count);
                else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) socket_error("recv");
                if (!header_read && received == distributed::message_header_size)
                {
                    std::size_t offset = 0;
                    const auto header = distributed::deserialize_header(receive_buffer, offset);
                    distributed::validate_header(header); // bound allocation before reading payload
                    receive_buffer.resize(distributed::message_header_size + header.payload_size);
                    header_read = true;
                }
            }
            if ((descriptors[0].revents & (POLLERR | POLLHUP | POLLNVAL)) ||
                (descriptors[1].revents & (POLLERR | POLLNVAL)))
                throw std::runtime_error("Ring connection failed");
        }
        std::size_t offset = 0;
        return distributed::deserialize_message(receive_buffer, offset);
    }
}

struct distributed::Ring::Impl
{
    RingOptions options;
    Socket outgoing;
    Socket incoming;
    std::uint64_t sequence = 0;
    bool failed = false;

    explicit Impl(const RingOptions &settings) : options(settings)
    {
        if (options.workers == 0 || options.rank >= options.workers || options.base_port == 0 ||
            options.workers > 65536U - options.base_port || options.timeout.count() <= 0 ||
            options.timeout.count() > std::numeric_limits<int>::max())
            throw std::invalid_argument("Invalid ring rank, worker count, port range, or timeout");
        if (options.hosts.empty()) options.hosts.assign(options.workers, "127.0.0.1");
        if (options.hosts.size() != options.workers)
            throw std::invalid_argument("Expected one host address per worker");
        for (const auto &host : options.hosts) (void)address(host, options.base_port);
        if ((options.incoming_fd >= 0) != (options.outgoing_fd >= 0) ||
            (options.incoming_fd >= 0 && options.incoming_fd == options.outgoing_fd))
            throw std::invalid_argument("Both distinct connected socket descriptors are required");
        if (options.workers == 1) return;

        const auto deadline = Clock::now() + options.timeout;
        if (options.incoming_fd >= 0)
        {
            outgoing = Socket(options.outgoing_fd);
            incoming = Socket(options.incoming_fd);
            configure(outgoing.get(), false);
            configure(incoming.get(), false);
        }
        else
        {
            Socket listener = listen_on(static_cast<std::uint16_t>(options.base_port + options.rank));
            const auto next = (options.rank + 1) % options.workers;
            outgoing = connect_to(options.hosts[next], static_cast<std::uint16_t>(options.base_port + next), deadline);
            incoming = accept_from(listener.get(), deadline);
        }
        Message hello;
        hello.header.type = MessageType::register_worker;
        hello.header.worker_id = static_cast<std::uint32_t>(options.rank);
        hello.header.element_count = static_cast<std::uint32_t>(options.workers);
        const auto peer = exchange(outgoing.get(), incoming.get(), hello, deadline);
        if (peer.header.type != MessageType::register_worker || peer.header.phase != Phase::setup ||
            peer.header.worker_id != previous() || peer.header.element_count != options.workers ||
            !peer.payload.empty())
            throw std::runtime_error("Ring registration mismatch");
    }

    std::size_t previous() const { return (options.rank + options.workers - 1) % options.workers; }
    Message transfer(Message message)
    {
        if (failed) throw std::runtime_error("Cannot reuse a failed ring");
        message.header.worker_id = static_cast<std::uint32_t>(options.rank);
        message.header.payload_size = static_cast<std::uint32_t>(message.payload.size());
        try
        {
            auto peer = exchange(outgoing.get(), incoming.get(), message, Clock::now() + options.timeout);
            if (peer.header.worker_id != previous()) throw std::runtime_error("Unexpected ring sender");
            return peer;
        }
        catch (...)
        {
            failed = true;
            outgoing = Socket();
            incoming = Socket();
            throw;
        }
    }
};

distributed::Ring::Ring(const RingOptions &options) : impl_(std::make_unique<Impl>(options)) {}
distributed::Ring::~Ring() = default;

void distributed::Ring::verify_configuration(const ByteBuffer &configuration)
{
    if (configuration.size() > max_payload_size) throw std::invalid_argument("Configuration too large");
    if (impl_->options.workers == 1) return;
    Message message;
    message.header.type = MessageType::ready;
    message.payload = configuration;
    const auto peer = impl_->transfer(std::move(message));
    if (peer.header.type != MessageType::ready || peer.header.phase != Phase::setup || peer.payload != configuration)
    {
        impl_->failed = true;
        throw std::runtime_error("Workers have different run configurations");
    }
}

void distributed::Ring::all_reduce_sum(std::vector<double> &values)
{
    if (impl_->failed) throw std::runtime_error("Cannot reuse a failed ring");
    const auto workers = impl_->options.workers;
    const auto rank = impl_->options.rank;
    // Balanced contiguous chunks, including empty chunks when length < workers.
    const auto chunk_size = [&](std::size_t chunk) { return values.size() / workers + (chunk < values.size() % workers ? 1U : 0U); };
    const auto chunk_offset = [&](std::size_t chunk) { return chunk * (values.size() / workers) + std::min(chunk, values.size() % workers); };
    if (chunk_size(0) > (max_payload_size - 16U) / sizeof(double))
        throw std::invalid_argument("Ring chunk exceeds maximum payload size");
    if (workers == 1) return;
    const auto sequence = impl_->sequence++;
    try
    {
        for (const Phase phase : {Phase::reduce_scatter, Phase::all_gather})
        {
            for (std::size_t step = 0; step + 1 < workers; ++step)
            {
                const auto send_chunk = (rank + workers - step + (phase == Phase::all_gather ? 1 : 0)) % workers;
                const auto receive_chunk = (send_chunk + workers - 1) % workers;
                Message message;
                message.header.type = phase == Phase::reduce_scatter ? MessageType::reduce_chunk : MessageType::gather_chunk;
                message.header.phase = phase;
                message.header.step = static_cast<std::uint32_t>(step);
                message.header.chunk_index = static_cast<std::uint32_t>(send_chunk);
                message.header.element_count = static_cast<std::uint32_t>(chunk_size(send_chunk));
                append_uint64(message.payload, sequence);
                append_uint64(message.payload, values.size());
                const auto start = chunk_offset(send_chunk);
                for (std::size_t i = 0; i < chunk_size(send_chunk); ++i) append_double(message.payload, values[start + i]);
                const auto peer = impl_->transfer(std::move(message));
                if (peer.header.type != (phase == Phase::reduce_scatter ? MessageType::reduce_chunk : MessageType::gather_chunk) ||
                    peer.header.phase != phase || peer.header.step != step || peer.header.chunk_index != receive_chunk ||
                    peer.header.element_count != chunk_size(receive_chunk) || peer.payload.size() != 16 + 8 * chunk_size(receive_chunk))
                    throw std::runtime_error("Unexpected ring phase, step, or chunk");
                std::size_t offset = 0;
                if (read_uint64(peer.payload, offset) != sequence || read_uint64(peer.payload, offset) != values.size())
                    throw std::runtime_error("Collective sequence or vector length mismatch");
                const auto destination = chunk_offset(receive_chunk);
                for (std::size_t i = 0; i < chunk_size(receive_chunk); ++i)
                {
                    const double value = read_double(peer.payload, offset);
                    if (phase == Phase::reduce_scatter) values[destination + i] += value;
                    else values[destination + i] = value;
                }
            }
        }
    }
    catch (...)
    {
        impl_->failed = true;
        impl_->outgoing = Socket();
        impl_->incoming = Socket();
        throw;
    }
}
