#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace distributed
{

    class Socket
    {
    public:
        // Creates an empty socket wrapper.
        Socket() noexcept;

        // Takes ownership of an existing socket file descriptor.
        explicit Socket(int file_descriptor) noexcept;

        ~Socket() noexcept;

        Socket(const Socket &) = delete;
        Socket &operator=(const Socket &) = delete;

        Socket(Socket &&other) noexcept;
        Socket &operator=(Socket &&other) noexcept;

        // Creates an IPv4 TCP socket.
        static Socket create_tcp();

        // Binds the socket to a local port.
        void bind(std::uint16_t port);

        // Marks the socket as a listening socket.
        void listen(int backlog = 16);

        // Accepts one incoming connection.
        Socket accept() const;

        // Connects to a remote TCP server.
        void connect(const std::string &host, std::uint16_t port);

        // Sends the entire byte buffer.
        void send_all(const std::vector<std::byte> &data) const;

        // Receives exactly byte_count bytes.
        std::vector<std::byte> receive_exact(std::size_t byte_count) const;

        // Enables non-blocking mode.
        void set_non_blocking();

        // Closes the socket if it is open.
        void close() noexcept;

        // Releases ownership without closing the descriptor.
        int release() noexcept;

        // Returns the underlying POSIX file descriptor.
        int file_descriptor() const noexcept;

        // Returns true when the wrapper owns an open descriptor.
        bool is_valid() const noexcept;

    private:
        int socket_fd_;
    };

} // namespace distributed
