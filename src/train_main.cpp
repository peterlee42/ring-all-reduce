#include "shallow_nn.hpp"
#include "dataset.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <iomanip>

int main(int argc, char *argv[])
{
    try
    {
        if (argc < 2)
        {
            throw std::runtime_error("Usage: train_main <path_to_csv>");
        }

        const std::string csv_path = argv[1];

        constexpr std::size_t input_dimension = 64;
        constexpr std::size_t hidden_dimension = 32;
        constexpr std::size_t output_dimension = 10;
        constexpr std::uint32_t seed = 42;
        constexpr float learning_rate = 0.05f;
        constexpr std::size_t maximum_rounds = 1500;
        constexpr double loss_threshold = 0.001;

        const Dataset dataset =
            Dataset::load_csv(csv_path, input_dimension);

        ShallowNetwork model(
            input_dimension,
            hidden_dimension,
            output_dimension,
            seed);

        for (std::size_t round = 0; round < maximum_rounds; ++round)
        {
            const TrainingStatistics statistics =
                model.compute_training_statistics(dataset);

            const double mean_loss =
                statistics.loss_sum /
                static_cast<double>(statistics.sample_count);

            if (!std::isfinite(mean_loss))
            {
                throw std::runtime_error("Training produced non-finite loss");
            }

            if (round % 10 == 0)
            {
                std::cout
                    << "round=" << round
                    << " loss=" << mean_loss
                    << '\n';
            }

            if (mean_loss < loss_threshold)
            {
                std::cout
                    << "training complete: loss threshold reached\n";
                break;
            }

            model.apply_gradient(
                statistics.gradient_sum,
                statistics.sample_count,
                learning_rate);
        }

        std::cout << "training complete\n";
        std::cout << "\nmodel parameters:\n";

        const std::vector<float> &parameters = model.parameters();

        constexpr std::size_t parameters_per_row = 4;
        constexpr int index_width = 5;
        constexpr int value_width = 14;
        constexpr int decimal_places = 6;

        std::cout << std::fixed << std::setprecision(decimal_places);

        for (std::size_t i = 0; i < parameters.size(); ++i)
        {
            std::cout
                << "param[" << std::setw(index_width) << i << "] = "
                << std::setw(value_width) << parameters[i];

            const bool end_of_row = (i + 1) % parameters_per_row == 0;
            const bool last_parameter = i + 1 == parameters.size();

            std::cout << (end_of_row || last_parameter ? "\n" : "    ");
        }

        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
