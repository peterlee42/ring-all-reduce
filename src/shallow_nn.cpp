#include "shallow_nn.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>

// ------------ PUBLIC METHODS ------------

ShallowNetwork::ShallowNetwork(
    std::size_t input_dimension,
    std::size_t hidden_dimension,
    std::size_t output_dimension,
    std::uint32_t seed)
    : input_dimension_(input_dimension),
      hidden_dimension_(hidden_dimension),
      output_dimension_(output_dimension),
      parameters_(hidden_dimension * input_dimension + hidden_dimension + hidden_dimension + 1, 0.0f)
{
    // Initialize random number generator with the provided seed (deterministic initialization)
    std::mt19937 rng(seed);

    const float standard_deviation = std::sqrt(2.0f / (static_cast<float>(input_dimension) + 1.0f));

    std::normal_distribution<float> dist(0.0f, standard_deviation);

    // Sample from normal distribution (mean=0, stddev=stddev) to initialize parameters
    for (float &param : parameters_)
    {
        param = dist(rng);
    }
}

float ShallowNetwork::predict_probability(
    const std::vector<float> &features) const
{
    if (features.size() != input_dimension_)
    {
        throw std::invalid_argument("feature dimension mismatch");
    }

    std::vector<float> z1(hidden_dimension_, 0.0f);
    std::vector<float> a1(hidden_dimension_, 0.0f);

    for (std::size_t h = 0; h < hidden_dimension_; ++h)
    {
        float value = parameters_[b1_index(h)];

        for (std::size_t d = 0; d < input_dimension_; ++d)
        {
            value += parameters_[w1_index(h, d)] * features[d];
        }

        z1[h] = value;
        a1[h] = value > 0.0f ? value : 0.0f;
    }

    float output_logit = parameters_[b2_index()];

    for (std::size_t h = 0; h < hidden_dimension_; ++h)
    {
        output_logit += parameters_[w2_index(h)] * a1[h];
    }

    return 1.0f / (1.0f + std::exp(-output_logit));
}

TrainingStatistics ShallowNetwork::compute_training_statistics(
    const Dataset &dataset) const
{
    TrainingStatistics result{
        std::vector<float>(parameter_count(), 0.0f),
        0.0,
        static_cast<std::uint64_t>(dataset.size())};

    constexpr float epsilon = 1.0e-7f;

    for (const Example &example : dataset.examples())
    {
        std::vector<float> z1(hidden_dimension_, 0.0f);
        std::vector<float> a1(hidden_dimension_, 0.0f);

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            float value = parameters_[b1_index(h)];

            for (std::size_t d = 0; d < input_dimension_; ++d)
            {
                value += parameters_[w1_index(h, d)] * example.features[d];
            }

            z1[h] = value;
            a1[h] = value > 0.0f ? value : 0.0f;
        }

        float output_logit = parameters_[b2_index()];

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            output_logit += parameters_[w2_index(h)] * a1[h];
        }

        const float probability =
            1.0f / (1.0f + std::exp(-output_logit));

        const float safe_probability =
            std::clamp(probability, epsilon, 1.0f - epsilon);

        result.loss_sum +=
            -static_cast<double>(example.label) *
                std::log(safe_probability) -
            static_cast<double>(1.0f - example.label) *
                std::log(1.0f - safe_probability);

        const float dz2 = probability - example.label;

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            result.gradient_sum[w2_index(h)] += dz2 * a1[h];
        }

        result.gradient_sum[b2_index()] += dz2;

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            const float da1 = dz2 * parameters_[w2_index(h)];
            const float dz1 = z1[h] > 0.0f ? da1 : 0.0f;

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
        1.0f / static_cast<float>(sample_count);

    for (std::size_t i = 0; i < parameters_.size(); ++i)
    {
        parameters_[i] -=
            learning_rate * gradient_sum[i] * inverse_count;
    }
}

const std::vector<float> &ShallowNetwork::parameters() const noexcept
{
    return parameters_;
}

std::size_t ShallowNetwork::input_dimension() const noexcept
{
    return input_dimension_;
}

std::size_t ShallowNetwork::hidden_dimension() const noexcept
{
    return hidden_dimension_;
}

std::size_t ShallowNetwork::parameter_count() const noexcept
{
    return parameters_.size();
}

// ------------ PRIVATE METHODS ------------

std::size_t ShallowNetwork::w1_index(std::size_t hidden, std::size_t input) const noexcept
{
    return hidden * input_dimension_ + input;
}

std::size_t ShallowNetwork::b1_index(std::size_t hidden) const noexcept
{
    return hidden_dimension_ * input_dimension_ + hidden;
}

std::size_t ShallowNetwork::w2_index(std::size_t hidden) const noexcept
{
    return hidden_dimension_ * input_dimension_ + hidden_dimension_ + hidden;
}

std::size_t ShallowNetwork::b2_index() const noexcept
{
    return hidden_dimension_ * input_dimension_ + hidden_dimension_ + hidden_dimension_;
}
