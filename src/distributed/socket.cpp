#pragma once

#include "distributed/socket.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

distributed::Socket::Socket()
    : socket_fd_(-1)
{
    // Create a TCP socket
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0)
    {
        throw std::runtime_error("Failed to create socket");
    }
}

explicit distributed::Socket::Socket(int file_descriptor) noexcept
    : socket_fd_(file_descriptor)
{
}

distributed::Socket::~Socket() noexcept
{
    close();
}

distributed::Socket::Socket(Socket &&other) noexcept
    : socket_fd_(other.socket_fd_)
{
    other.socket_fd_ = -1;
}

distributed::Socket &distributed::Socket::operator=(Socket &&other) noexcept
{
    if (this != &other)
    {
        close();
        socket_fd_ = other.socket_fd_;
        other.socket_fd_ = -1;
    }
    return *this;
}

distributed::Socket distributed::Socket::create_tcp()
{
    return Socket();
}

void distributed::Socket::bind(std::uint16_t port)
{
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (::bind(socket_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        throw std::runtime_error("Failed to bind socket");
    }
}

void distributed::Socket::listen(int backlog)
{
    if (::listen(socket_fd_, backlog) < 0)
    {
        throw std::runtime_error("Failed to listen on socket");
    }
}

distributed::Socket distributed::Socket::accept() const
{
    int client_fd = ::accept(socket_fd_, nullptr, nullptr);
    if (client_fd < 0)
    {
        throw std::runtime_error("Failed to accept connection");
    }
    return Socket(client_fd);
}

void distributed::Socket::connect(const std::string &host, std::uint16_t port)
{
    addrinfo hints{}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0)
    {
        throw std::runtime_error("Failed to resolve host");
    }

    if (::connect(socket_fd_, res->ai_addr, res->ai_addrlen) < 0)
    {
        freeaddrinfo(res);
        throw std::runtime_error("Failed to connect to server");
    }

    freeaddrinfo(res);
}

void distributed::Socket::send_all(const std::vector<std::byte> &data) const
{
    size_t total_sent = 0;
    while (total_sent < data.size())
    {
        ssize_t sent = ::send(socket_fd_, data.data() + total_sent, data.size() - total_sent, 0);
        if (sent < 0)
        {
            throw std::runtime_error("Failed to send data");
        }
        total_sent += sent;
    }
}

std::vector<std::byte> distributed::Socket::receive_exact(std::size_t byte_count) const
{
    std::vector<std::byte> buffer(byte_count);
    size_t total_received = 0;
    while (total_received < byte_count)
    {
        ssize_t received = ::recv(socket_fd_, buffer.data() + total_received, byte_count - total_received, 0);
        if (received < 0)
        {
            throw std::runtime_error("Failed to receive data");
        }
        total_received += received;
    }
    return buffer;
}

void distributed::Socket::set_non_blocking()
{
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags < 0)
    {
        throw std::runtime_error("Failed to get socket flags");
    }
    if (fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        throw std::runtime_error("Failed to set socket to non-blocking mode");
    }
}

void distributed::Socket::close() noexcept
{
    if (socket_fd_ >= 0)
    {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

int distributed::Socket::release() noexcept
{
    int fd = socket_fd_;
    socket_fd_ = -1;
    return fd;
}

int distributed::Socket::file_descriptor() const noexcept
{
    return socket_fd_;
}

bool distributed::Socket::is_valid() const noexcept
{
    return socket_fd_ >= 0;
}
