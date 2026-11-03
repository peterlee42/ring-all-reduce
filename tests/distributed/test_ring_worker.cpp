#include "distributed/ring.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

int main(int argc, char **argv)
{
    try
    {
        if (argc != 5 && argc != 7) throw std::invalid_argument("Expected rank workers port mode");
        distributed::RingOptions options;
        options.rank = std::stoul(argv[1]);
        options.workers = std::stoul(argv[2]);
        options.base_port = static_cast<std::uint16_t>(std::stoul(argv[3]));
        const std::string mode = argv[4];
        options.timeout = std::chrono::milliseconds(mode == "success" ? 10000 : 1200);
        if (argc == 7)
        {
            options.incoming_fd = std::stoi(argv[5]);
            options.outgoing_fd = std::stoi(argv[6]);
        }
        distributed::Ring ring(options);
        if (mode == "disconnect" && options.rank == 1) return 0;
        if (mode == "stall" && options.rank == 1)
        {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            return 0;
        }
        if (mode == "configuration")
        {
            ring.verify_configuration({static_cast<std::byte>(options.rank)});
            throw std::runtime_error("Configuration mismatch was accepted");
        }
        ring.verify_configuration({std::byte{1}, std::byte{2}});
        if (mode != "success")
        {
            std::vector<double> values(mode == "length" ? options.rank + 3 : 3, 1.0);
            ring.all_reduce_sum(values);
            throw std::runtime_error("Expected collective failure");
        }
        // Sizes cross chunk boundaries, include empty chunks, and exceed socket buffers.
        for (const std::size_t size : {0U, 1U, 2U, 7U, 17U, 1000003U})
        {
            for (int round = 0; round < 3; ++round)
            {
                std::vector<double> values(size);
                for (std::size_t i = 0; i < size; ++i)
                    values[i] = (options.rank + 1.0) * (static_cast<double>(i % 13) - 6.0) + round;
                ring.all_reduce_sum(values);
                const double workers = static_cast<double>(options.workers);
                for (std::size_t i = 0; i < size; ++i)
                {
                    const double expected = workers * (workers + 1.0) / 2.0 * (static_cast<double>(i % 13) - 6.0) + workers * round;
                    if (values[i] != expected) throw std::runtime_error("Wrong reduced value");
                }
            }
        }
        std::vector<double> done{1.0};
        ring.all_reduce_sum(done);
        std::cout << "verified rank=" << options.rank << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
