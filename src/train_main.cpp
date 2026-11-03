#include "distributed/ring.hpp"
#include "ml/dataset.hpp"
#include "ml/shallow_nn.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
    struct Options
    {
        distributed::RingOptions ring;
        std::string csv;
        std::string output;
        std::size_t input = 64, hidden = 64, classes = 10, rounds = 500, elements = 17;
        std::uint32_t seed = 42;
        float learning_rate = 0.05f;
        double threshold = 0.01;
        bool demo = false;
    };

    void usage()
    {
        std::cout << "Usage: train [DATA.csv | --demo] [options]\n"
                  << "  --rank N --workers N --base-port N --hosts IPv4,IPv4,...\n"
                  << "  --input N --hidden N --classes N --seed N --rounds N\n"
                  << "  --lr RATE --threshold LOSS --timeout-ms N\n"
                  << "  --elements N       Vector length for the verified all-reduce demo\n"
                  << "  --output PATH      Write final parameters as JSON (rank zero only)\n"
                  << "CSV: header, feature columns, then a zero-based integer label.\n";
    }

    std::size_t integer(const std::string &value)
    {
        std::size_t result = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
            throw std::invalid_argument("Invalid nonnegative integer: " + value);
        return result;
    }

    double number(const std::string &value)
    {
        std::size_t consumed = 0;
        const auto result = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(result)) throw std::invalid_argument("Invalid number: " + value);
        return result;
    }

    Options parse(int argc, char **argv)
    {
        Options options;
        for (int i = 1; i < argc; ++i)
        {
            const std::string key = argv[i];
            if (key == "--demo") { options.demo = true; continue; }
            if (!key.starts_with("--"))
            {
                if (!options.csv.empty()) throw std::invalid_argument("Only one dataset path is allowed");
                options.csv = key;
                continue;
            }
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--rank") options.ring.rank = integer(value);
            else if (key == "--workers") options.ring.workers = integer(value);
            else if (key == "--base-port")
            {
                const auto port = integer(value);
                if (port == 0 || port > 65535) throw std::invalid_argument("Invalid base port");
                options.ring.base_port = static_cast<std::uint16_t>(port);
            }
            else if (key == "--hosts")
            {
                std::istringstream hosts(value);
                std::string host;
                while (std::getline(hosts, host, ',')) options.ring.hosts.push_back(host);
                if (value.empty() || value.back() == ',') throw std::invalid_argument("Invalid host list");
            }
            else if (key == "--timeout-ms")
            {
                const auto timeout = integer(value);
                if (timeout == 0 || timeout > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("Invalid timeout");
                options.ring.timeout = std::chrono::milliseconds(timeout);
            }
            else if (key == "--incoming-fd" || key == "--outgoing-fd")
            {
                const auto fd = integer(value);
                if (fd > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("Invalid socket descriptor");
                if (key == "--incoming-fd") options.ring.incoming_fd = static_cast<int>(fd);
                else options.ring.outgoing_fd = static_cast<int>(fd);
            }
            else if (key == "--input") options.input = integer(value);
            else if (key == "--hidden") options.hidden = integer(value);
            else if (key == "--classes") options.classes = integer(value);
            else if (key == "--rounds") options.rounds = integer(value);
            else if (key == "--elements") options.elements = integer(value);
            else if (key == "--seed")
            {
                const auto seed = integer(value);
                if (seed > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("Seed is too large");
                options.seed = static_cast<std::uint32_t>(seed);
            }
            else if (key == "--lr") options.learning_rate = static_cast<float>(number(value));
            else if (key == "--threshold") options.threshold = number(value);
            else if (key == "--output") options.output = value;
            else throw std::invalid_argument("Unknown option: " + key);
        }
        if (options.demo == !options.csv.empty()) throw std::invalid_argument("Specify a dataset path or --demo");
        if (options.rounds == 0 || !std::isfinite(options.learning_rate) || options.learning_rate <= 0 || options.threshold < 0)
            throw std::invalid_argument("Rounds and learning rate must be positive; threshold must be nonnegative");
        // Bound allocations and keep the gradient within the collective payload limit.
        if (options.input == 0 || options.input > 4096 || options.hidden == 0 || options.hidden > 4096 ||
            options.classes < 2 || options.classes > 4096 || options.elements > 10000000)
            throw std::invalid_argument("Dimensions out of range (input/hidden: 1..4096, classes: 2..4096, elements: 0..10000000)");
        return options;
    }

    distributed::ByteBuffer configuration(const Options &options, const ml::Dataset *dataset)
    {
        distributed::ByteBuffer bytes;
        for (const auto value : {options.input, options.hidden, options.classes, options.rounds, options.elements})
            distributed::append_uint64(bytes, value);
        distributed::append_uint32(bytes, options.seed);
        distributed::append_float(bytes, options.learning_rate);
        distributed::append_double(bytes, options.threshold);
        distributed::append_uint16(bytes, options.demo ? 1 : 0);
        if (dataset)
        {
            // Stable content fingerprint: detect ranks loading different full CSVs.
            // This is a consistency check, not a cryptographic authentication tag.
            std::uint64_t hash = 14695981039346656037ULL;
            for (const auto &example : dataset->examples())
            {
                distributed::ByteBuffer row;
                for (const float value : example.features) distributed::append_float(row, value);
                distributed::append_uint64(row, example.label);
                for (const auto byte : row)
                {
                    hash ^= std::to_integer<std::uint8_t>(byte);
                    hash *= 1099511628211ULL;
                }
            }
            distributed::append_uint64(bytes, dataset->size());
            distributed::append_uint64(bytes, hash);
        }
        return bytes;
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        if (argc == 1) { usage(); return 1; }
        const auto options = parse(argc, argv);
        std::cout << std::setprecision(10) << std::unitbuf;
        std::optional<ml::Dataset> loaded;
        if (!options.demo) loaded = ml::Dataset::load_csv(options.csv, options.input);
        distributed::Ring ring(options.ring);
        ring.verify_configuration(configuration(options, loaded ? &*loaded : nullptr));
        if (options.demo)
        {
            for (std::size_t round = 0; round < options.rounds; ++round)
            {
                std::vector<double> values(options.elements);
                for (std::size_t i = 0; i < values.size(); ++i) values[i] = options.ring.rank + 1.0 + static_cast<double>(i) + static_cast<double>(round);
                ring.all_reduce_sum(values);
                const double workers = static_cast<double>(options.ring.workers);
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    const double expected = workers * (workers + 1.0) / 2.0 + workers * (static_cast<double>(i) + static_cast<double>(round));
                    if (values[i] != expected) throw std::runtime_error("All-reduce verification failed at element " + std::to_string(i));
                }
                if (round + 1 == options.rounds)
                    std::cout << "rank=" << options.ring.rank << " all_reduce=ok workers=" << options.ring.workers
                              << " elements=" << values.size() << " rounds=" << options.rounds << '\n';
            }
            return 0;
        }

        const auto &dataset = *loaded;
        const auto local = dataset.shard(options.ring.rank, options.ring.workers);
        ml::ShallowNetwork model(options.input, options.hidden, options.classes, options.seed);
        std::cout << "rank=" << options.ring.rank << " samples=" << local.size() << " parameters=" << model.parameter_count() << '\n';
        std::size_t updates = 0;
        const auto statistics = [&]()
        {
            const auto local_statistics = model.compute_training_statistics(local);
            std::vector<double> total(local_statistics.gradient_sum.begin(), local_statistics.gradient_sum.end());
            total.push_back(local_statistics.loss_sum);
            total.push_back(static_cast<double>(local_statistics.sample_count));
            ring.all_reduce_sum(total);
            const double count = total.back();
            total.pop_back();
            const double loss = total.back();
            total.pop_back();
            if (!std::isfinite(count) || count <= 0 || count > 9007199254740991.0 || std::floor(count) != count || !std::isfinite(loss))
                throw std::runtime_error("Invalid global training statistics");
            ml::TrainingStatistics result{{}, loss, static_cast<std::uint64_t>(count)};
            result.gradient_sum.assign(total.begin(), total.end());
            return result;
        };
        for (std::size_t round = 0; round < options.rounds; ++round)
        {
            const auto total = statistics();
            const double loss = total.loss_sum / static_cast<double>(total.sample_count);
            if (options.ring.rank == 0 && round % 10 == 0) std::cout << "round=" << round << " loss=" << loss << '\n';
            if (loss <= options.threshold) break;
            model.apply_gradient(total.gradient_sum, total.sample_count, options.learning_rate);
            ++updates;
        }
        const auto final = statistics();
        if (options.ring.rank == 0)
        {
            // Every process has the full CSV, but only its shard contributes gradients.
            std::size_t correct = 0;
            for (const auto &example : dataset.examples()) correct += model.predict_class(example.features) == example.label ? 1 : 0;
            const double loss = final.loss_sum / static_cast<double>(final.sample_count);
            std::cout << "training complete updates=" << updates << " loss=" << loss
                      << " accuracy=" << static_cast<double>(correct) / static_cast<double>(dataset.size())
                      << " reason=" << (loss <= options.threshold ? "loss_threshold" : "maximum_rounds") << '\n';
            if (!options.output.empty())
            {
                std::ofstream file(options.output);
                file << std::setprecision(std::numeric_limits<float>::max_digits10);
                file << "{\"input\":" << options.input << ",\"hidden\":" << options.hidden << ",\"classes\":" << options.classes << ",\"parameters\":[";
                for (std::size_t i = 0; i < model.parameters().size(); ++i)
                {
                    if (i) file << ',';
                    file << model.parameters()[i];
                }
                file << "]}\n";
                file.close();
                if (!file) throw std::runtime_error("Failed writing model: " + options.output);
            }
        }
        // Keep peers alive until rank zero has finished reporting/writing.
        std::vector<double> done{1.0};
        ring.all_reduce_sum(done);
        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
