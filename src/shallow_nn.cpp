#include "shallow_nn.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>
#include <vector>

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

    // Xavier initialization for weights and biases
    const float standard_deviation = std::sqrt(2.0f / (static_cast<float>(input_dimension) + 1.0f));
    std::uniform_real_distribution<float> dist(-standard_deviation, standard_deviation);

    // Initialize weights and biases
    for (float &param : parameters_)
    {
        param = dist(rng);
    }
}

// ------------ Inference ------------
std::vector<float> ShallowNetwork::predict_probabilities(const std::vector<float> &features) const
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
        a1[h] = value > 0.0f ? value : 0.0f; // ReLU activation
    }

    std::vector<float> logits(output_dimension_, 0.0f);

    for (std::size_t o = 0; o < output_dimension_; ++o)
    {
        float value = parameters_[b2_index(o)];

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            value += parameters_[w2_index(o, h)] * a1[h];
        }

        logits[o] = value; // Linear activation for output layer
    }

    std::vector<float> probabilities(output_dimension_);

    const float maximum_logit = *std::max_element(logits.begin(), logits.end());

    float exponential_sum = 0.0f;

    for (std::size_t o = 0; o < output_dimension_; ++o)
    {
        probabilities[o] = std::exp(logits[o] - maximum_logit); // Subtract maximum logit for numerical stability
        exponential_sum += probabilities[o];
    }

    for (float &probability : probabilities)
    {
        probability /= exponential_sum;
    }

    return probabilities;
}

// ------------ Training ------------
TrainingStatistics ShallowNetwork::compute_training_statistics(
    const Dataset &dataset) const
{
    if (dataset.input_dimension() != input_dimension_)
    {
        throw std::invalid_argument("dataset input dimension mismatch");
    }

    if (dataset.size() == 0)
    {
        throw std::invalid_argument("dataset is empty");
    }

    TrainingStatistics statistics{
        std::vector<float>(parameters_.size(), 0.0f),
        0.0,
        static_cast<std::uint64_t>(dataset.size())};

    constexpr float epsilon = 1.0e-8f;

    for (const Example &example : dataset.examples())
    {
        if (example.features.size() != input_dimension_)
        {
            throw std::invalid_argument("example feature dimension mismatch");
        }

        if (example.label >= output_dimension_)
        {
            throw std::invalid_argument("example label out of range");
        }

        // --------------------------------------------------------------
        // Forward Pass
        // --------------------------------------------------------------
        const ForwardPass pass = forward(example.features);

        // --------------------------------------------------------------
        // Cross-entropy loss
        // --------------------------------------------------------------
    };
}

void ShallowNetwork::apply_gradient(
    const std::vector<float> &gradient_sum,
    std::uint64_t sample_count,
    float learning_rate)
{
    return;
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

std::size_t ShallowNetwork::output_dimension() const noexcept
{
    return output_dimension_;
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
    const std::size_t w1_size =
        hidden_dimension_ * input_dimension_;

    return w1_size + hidden;
}

std::size_t ShallowNetwork::w2_index(std::size_t output, std::size_t hidden) const noexcept
{
    const std::size_t w1_size =
        hidden_dimension_ * input_dimension_;

    const std::size_t b1_size =
        hidden_dimension_;

    return w1_size + b1_size + output * hidden_dimension_ + hidden;
}

std::size_t ShallowNetwork::b2_index(std::size_t output) const noexcept
{
    const std::size_t w1_size =
        hidden_dimension_ * input_dimension_;

    const std::size_t b1_size =
        hidden_dimension_;

    const std::size_t w2_size =
        output_dimension_ * hidden_dimension_;

    return w1_size + b1_size + w2_size + output;
}

ShallowNetwork::ForwardPass ShallowNetwork::forward(const std::vector<float> &features) const
{
    ShallowNetwork::ForwardPass pass;

    pass.hidden_pre_activations.resize(hidden_dimension_);
    pass.hidden_activations.resize(hidden_dimension_);
    pass.logits.resize(output_dimension_);
    pass.probabilities.resize(output_dimension_);

    for (std::size_t h = 0; h < hidden_dimension_; ++h)
    {
        float value{parameters_[b1_index(h)]};

        for (std::size_t d = 0; d < input_dimension_; ++d)
        {
            value += parameters_[w1_index(h, d)] * features[d];
        }

        pass.hidden_pre_activations[h] = value;
        pass.hidden_activations[h] = value > 0.0f ? value : 0.0f; // ReLU activation
    }

    for (std::size_t o = 0; o < output_dimension_; ++o)
    {
        float value = parameters_[b2_index(o)];

        for (std::size_t h = 0; h < hidden_dimension_; ++h)
        {
            value += parameters_[w2_index(o, h)] * pass.hidden_activations[h];
        }

        pass.logits[o] = value; // Linear activation for output layer
    }

    const float maximum_logit = *std::max_element(pass.logits.begin(), pass.logits.end());

    float exponential_sum = 0.0f;

    for (std::size_t o = 0; o < output_dimension_; ++o)
    {
        pass.probabilities[o] = std::exp(pass.logits[o] - maximum_logit); // Subtract maximum logit for numerical stability
        exponential_sum += pass.probabilities[o];
    }

    for (float &probability : pass.probabilities)
    {
        probability /= exponential_sum;
    }

    return pass;
}
