#include "ml/shallow_nn.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>
#include <vector>

// ------------ PUBLIC METHODS ------------

ml::ShallowNetwork::ShallowNetwork(
    std::size_t input_dimension,
    std::size_t hidden_dimension,
    std::size_t output_dimension,
    std::uint32_t seed)
    : input_dimension_(input_dimension),
      hidden_dimension_(hidden_dimension),
      output_dimension_(output_dimension),
      parameters_(hidden_dimension * input_dimension + hidden_dimension + hidden_dimension * output_dimension + output_dimension, 0.0f)
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
std::vector<float> ml::ShallowNetwork::predict_probabilities(const std::vector<float> &features) const
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

//------------ Inference ------------
std::size_t ml::ShallowNetwork::predict_class(const std::vector<float> &features) const
{
    const std::vector<float> probabilities = predict_probabilities(features);
    return std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
}

// ------------ Training ------------
ml::TrainingStatistics ml::ShallowNetwork::compute_training_statistics(
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

    ml::TrainingStatistics statistics{
        std::vector<float>(parameters_.size(), 0.0f),
        0.0,
        static_cast<std::uint64_t>(dataset.size())};

    constexpr float epsilon = 1.0e-7f;

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
        // Cross-entropy loss (multi-class classification)
        // --------------------------------------------------------------
        const float correct_class_probability = std::max(pass.probabilities[example.label], epsilon);
        statistics.loss_sum -= std::log(static_cast<double>(correct_class_probability));

        // --------------------------------------------------------------
        // Output-layer delta
        //
        // For softmax combined with cross-entropy:
        //
        // delta = probabilities - one_hot_target
        // --------------------------------------------------------------
        std::vector<float> output_delta = pass.probabilities;
        output_delta[example.label] -= 1.0f;

        // --------------------------------------------------------------
        // Gradients for W2 and b2
        // --------------------------------------------------------------
        for (std::size_t output = 0; output < output_dimension_; ++output)
        {
            for (std::size_t hidden = 0; hidden < hidden_dimension_; ++hidden)
            {
                statistics.gradient_sum[w2_index(output, hidden)] += output_delta[output] * pass.hidden_activations[hidden];
            }

            statistics.gradient_sum[b2_index(output)] += output_delta[output];
        }

        // --------------------------------------------------------------
        // Backpropagate into the hidden layer
        // --------------------------------------------------------------
        for (std::size_t hidden = 0; hidden < hidden_dimension_; ++hidden)
        {
            float hidden_gradient = 0.0f;

            for (std::size_t output = 0; output < output_dimension_; ++output)
            {
                hidden_gradient += output_delta[output] * parameters_[w2_index(output, hidden)];
            }

            // Derivative of ReLU.
            const float hidden_delta = pass.hidden_pre_activations[hidden] > 0.0f ? hidden_gradient : 0.0f;

            // Gradient for the hidden bias.
            statistics.gradient_sum[b1_index(hidden)] += hidden_delta;

            // Gradient for the input-to-hidden weights.
            for (std::size_t input = 0; input < input_dimension_; ++input)
            {
                statistics.gradient_sum[w1_index(hidden, input)] += hidden_delta * example.features[input];
            }
        }
    }

    if (!std::isfinite(statistics.loss_sum))
    {
        throw std::runtime_error("Training produced a non-finite loss");
    }

    for (const float gradient : statistics.gradient_sum)
    {
        if (!std::isfinite(gradient))
        {
            throw std::runtime_error("Training produced a non-finite gradient");
        }
    }

    return statistics;
}

void ml::ShallowNetwork::apply_gradient(
    const std::vector<float> &gradient_sum,
    std::uint64_t sample_count,
    float learning_rate)
{
    if (gradient_sum.size() != parameters_.size())
    {
        throw std::invalid_argument("gradient dimension mismatch");
    }

    if (sample_count == 0)
    {
        throw std::invalid_argument("sample count is zero");
    }

    const float scale = learning_rate / static_cast<float>(sample_count);

    for (std::size_t i = 0; i < parameters_.size(); ++i)
    {
        parameters_[i] -= scale * gradient_sum[i];
    }
}

const std::vector<float> &ml::ShallowNetwork::parameters() const noexcept
{
    return parameters_;
}

std::size_t ml::ShallowNetwork::input_dimension() const noexcept
{
    return input_dimension_;
}

std::size_t ml::ShallowNetwork::hidden_dimension() const noexcept
{
    return hidden_dimension_;
}

std::size_t ml::ShallowNetwork::output_dimension() const noexcept
{
    return output_dimension_;
}

std::size_t ml::ShallowNetwork::parameter_count() const noexcept
{
    return parameters_.size();
}

// ------------ PRIVATE METHODS ------------
std::size_t ml::ShallowNetwork::w1_index(std::size_t hidden, std::size_t input) const noexcept
{
    return hidden * input_dimension_ + input;
}

std::size_t ml::ShallowNetwork::b1_index(std::size_t hidden) const noexcept
{
    const std::size_t w1_size =
        hidden_dimension_ * input_dimension_;

    return w1_size + hidden;
}

std::size_t ml::ShallowNetwork::w2_index(std::size_t output, std::size_t hidden) const noexcept
{
    const std::size_t w1_size =
        hidden_dimension_ * input_dimension_;

    const std::size_t b1_size =
        hidden_dimension_;

    return w1_size + b1_size + output * hidden_dimension_ + hidden;
}

std::size_t ml::ShallowNetwork::b2_index(std::size_t output) const noexcept
{
    const std::size_t w1_size =
        hidden_dimension_ * input_dimension_;

    const std::size_t b1_size =
        hidden_dimension_;

    const std::size_t w2_size =
        output_dimension_ * hidden_dimension_;

    return w1_size + b1_size + w2_size + output;
}

ml::ShallowNetwork::ForwardPass ml::ShallowNetwork::forward(const std::vector<float> &features) const
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
