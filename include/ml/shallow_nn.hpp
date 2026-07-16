#pragma once

#include "dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml
{
    struct TrainingStatistics
    {
        std::vector<float> gradient_sum;
        double loss_sum;
        std::uint64_t sample_count;
    };

    class ShallowNetwork
    {
    public:
        ShallowNetwork(
            std::size_t input_dimension,
            std::size_t hidden_dimension,
            std::size_t output_dimension,
            std::uint32_t seed);

        std::vector<float> predict_probabilities(
            const std::vector<float> &features) const;

        std::size_t predict_class(
            const std::vector<float> &features) const;

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
        struct ForwardPass
        {
            std::vector<float> hidden_pre_activations;
            std::vector<float> hidden_activations;
            std::vector<float> logits;
            std::vector<float> probabilities;
        };

        ForwardPass forward(
            const std::vector<float> &features) const;

        std::size_t w1_index(
            std::size_t hidden,
            std::size_t input) const noexcept;

        std::size_t b1_index(
            std::size_t hidden) const noexcept;

        std::size_t w2_index(
            std::size_t output,
            std::size_t hidden) const noexcept;

        std::size_t b2_index(
            std::size_t output) const noexcept;

        std::size_t input_dimension_;
        std::size_t hidden_dimension_;
        std::size_t output_dimension_;
        std::vector<float> parameters_;
    };
}
