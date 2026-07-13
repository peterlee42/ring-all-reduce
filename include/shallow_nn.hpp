#pragma once

#include "dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct TrainingStatistics
{
    std::vector<float> gradient_sum;
    double loss_sum;
    std::uint64_t sample_count;
};

class ShallowNetwork
{
public:
    ShallowNetwork(std::size_t input_dimension, std::size_t hidden_dimension, std::size_t output_dimension, std::uint32_t seed);

    float predict_probability(const std::vector<float> &features) const;

    TrainingStatistics compute_training_statistics(
        const Dataset &dataset) const;

    void apply_gradient(
        const std::vector<float> &gradient_sum,
        std::uint64_t sample_count,
        float learning_rate);

    const std::vector<float> &parameters() const noexcept;

    std::size_t input_dimension() const noexcept;
    std::size_t hidden_dimension() const noexcept;
    std::size_t output_dimension() const noexcept;
    std::size_t parameter_count() const noexcept;

private:
    std::size_t w1_index(std::size_t hidden, std::size_t input) const noexcept;
    std::size_t b1_index(std::size_t hidden) const noexcept;
    std::size_t w2_index(std::size_t hidden) const noexcept;
    std::size_t b2_index() const noexcept;

    std::size_t input_dimension_;
    std::size_t hidden_dimension_;
    std::size_t output_dimension_;
    std::vector<float> parameters_;
};
