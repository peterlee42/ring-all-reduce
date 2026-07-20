#pragma once

#include "distributed/socket.hpp"
#include "distributed/worker_registration.hpp"

#include <cstdint>
#include <vector>

namespace distributed
{
    class Coordinator
    {
    public:
        Coordinator(std::uint16_t port, std::uint32_t expected_workers);

        void run();

    private:
        void accept_workers();
        void receive_registration(Socket &connection);
        void assign_worker_ids();
        void send_ring_configuration();
        void wait_until_ready();
        void broadcast_start();

        std::uint16_t port_;
        std::uint32_t expected_workers_;
        Socket listening_socket_;
        std::vector<RegisteredWorker> workers_;
    };
}
