#include "ml/shallow_nn.hpp"
#include "ml/dataset.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

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
        constexpr std::size_t hidden_dimension = 64;
        constexpr std::size_t output_dimension = 10;
        constexpr std::uint32_t seed = 42;
        constexpr float learning_rate = 0.05f;
        constexpr std::size_t maximum_rounds = 500;
        constexpr double loss_threshold = 0.01;

        const ml::Dataset dataset =
            ml::Dataset::load_csv(csv_path, input_dimension);

        ml::ShallowNetwork model(
            input_dimension,
            hidden_dimension,
            output_dimension,
            seed);

        for (std::size_t round = 0; round < maximum_rounds; ++round)
        {
            const ml::TrainingStatistics statistics =
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

        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
