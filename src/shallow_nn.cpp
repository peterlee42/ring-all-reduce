#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "shallow_nn.hpp"

TrainingStatistics ShallowNetwork::compute_training_statistics(
    const Dataset &dataset) const
{
    TrainingStatistics result{
        std::vector<float>(parameter_count(), 0.0F),
        0.0,
        static_cast<std::uint64_t>(dataset.size())};

    constexpr float epsilon = 1.0e-7F;

    for (const Example &example : dataset.examples())
    {
        std::vector<float> z1(hidden_dimension_, 0.0F);
        std::vector<float> a1(hidden_dimension_, 0.0F);

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            float value = parameters_[b1_index(h)];

            for (std::size_t d = 0; d < input_dimension_; ++d)
            {
                value += parameters_[w1_index(h, d)] * example.features[d];
            }

            z1[h] = value;
            a1[h] = value > 0.0F ? value : 0.0F;
        }

        float output_logit = parameters_[b2_index()];

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            output_logit += parameters_[w2_index(h)] * a1[h];
        }

        const float probability =
            1.0F / (1.0F + std::exp(-output_logit));

        const float safe_probability =
            std::clamp(probability, epsilon, 1.0F - epsilon);

        result.loss_sum +=
            -static_cast<double>(example.label) *
                std::log(safe_probability) -
            static_cast<double>(1.0F - example.label) *
                std::log(1.0F - safe_probability);

        const float dz2 = probability - example.label;

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            result.gradient_sum[w2_index(h)] += dz2 * a1[h];
        }

        result.gradient_sum[b2_index()] += dz2;

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            const float da1 = dz2 * parameters_[w2_index(h)];
            const float dz1 = z1[h] > 0.0F ? da1 : 0.0F;

            result.gradient_sum[b1_index(h)] += dz1;

            for (std::size_t d = 0; d < input_dimension_; ++d)
            {
                result.gradient_sum[w1_index(h, d)] +=
                    dz1 * example.features[d];
            }
        }
    }

    return result;
}

void ShallowNetwork::apply_gradient(
    const std::vector<float> &gradient_sum,
    std::uint64_t sample_count,
    float learning_rate)
{
    if (gradient_sum.size() != parameters_.size())
    {
        throw std::invalid_argument("Gradient dimension mismatch");
    }

    if (sample_count == 0)
    {
        throw std::invalid_argument("Sample count must be greater than zero");
    }

    const float inverse_count =
        1.0F / static_cast<float>(sample_count);

    for (std::size_t i = 0; i < parameters_.size(); ++i)
    {
        parameters_[i] -=
            learning_rate * gradient_sum[i] * inverse_count;
    }
}
