#pragma once

#include <cstdint>
#include <string>

namespace distributed
{
    class RegisteredWorker
    {
    public:
        RegisteredWorker(std::uint32_t id, const std::string &host, std::uint16_t port)
            : id_(id), host_(host), port_(port) {}

        std::uint32_t id() const { return id_; }
        const std::string &host() const { return host_; }
        std::uint16_t port() const { return port_; }

    private:
        std::uint32_t id_;
        std::string host_;
        std::uint16_t port_;
    };
}
