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

static distributed::Socket distributed::Socket::create_tcp()
{
    return Socket();
}
