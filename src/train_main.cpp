#include "dataset.hpp"
#include "shallow_nn.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main()
{
    try
    {
        constexpr std::size_t input_dimension = 2;
        constexpr std::size_t hidden_dimension = 8;
        constexpr std::uint32_t seed = 42;
        constexpr float learning_rate = 0.01F;
        constexpr std::size_t maximum_rounds = 1000;
        constexpr double loss_threshold = 0.05;

        const Dataset dataset =
            Dataset::load_csv("data/sample.csv", input_dimension);

        ShallowNetwork model(
            input_dimension,
            hidden_dimension,
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

            if (mean_loss <= loss_threshold)
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
