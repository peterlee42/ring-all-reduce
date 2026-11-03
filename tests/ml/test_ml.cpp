#include "ml/dataset.hpp"
#include "ml/shallow_nn.hpp"

#include "test_support.hpp"
#include <cmath>
#include <limits>

namespace
{
    ml::Dataset data()
    {
        return ml::Dataset::from_examples({{{-1.0f, -1.0f}, 0}, {{-0.5f, -1.0f}, 0},
                                          {{1.0f, 1.0f}, 1}, {{0.5f, 1.0f}, 1}, {{0.8f, 0.8f}, 1}}, 2);
    }
}

TEST_CASE(Training, ShardSumsMatchFullDatasetIncludingEmptyShard)
{
    ml::ShallowNetwork model(2, 8, 2, 42);
    const auto dataset = data();
    const auto full = model.compute_training_statistics(dataset);
    for (const auto workers : {2U, 3U, 7U})
    {
        std::vector<float> gradient(model.parameter_count(), 0.0f);
        double loss = 0;
        std::uint64_t samples = 0;
        for (std::size_t rank = 0; rank < workers; ++rank)
        {
            const auto part = model.compute_training_statistics(dataset.shard(rank, workers));
            samples += part.sample_count;
            loss += part.loss_sum;
            for (std::size_t i = 0; i < gradient.size(); ++i) gradient[i] += part.gradient_sum[i];
        }
        CHECK_EQ(samples, full.sample_count);
        CHECK_NEAR(loss, full.loss_sum, 1e-7);
        for (std::size_t i = 0; i < gradient.size(); ++i) CHECK_NEAR(gradient[i], full.gradient_sum[i], 1e-6);
    }
}

TEST_CASE(Training, LearnsAndProducesNormalizedProbabilities)
{
    ml::ShallowNetwork model(2, 8, 2, 42);
    const auto dataset = data();
    const double initial = model.compute_training_statistics(dataset).loss_sum;
    for (int step = 0; step < 100; ++step)
    {
        const auto stats = model.compute_training_statistics(dataset);
        model.apply_gradient(stats.gradient_sum, stats.sample_count, 0.1f);
    }
    CHECK_LT(model.compute_training_statistics(dataset).loss_sum, initial * 0.1);
    for (const auto &example : dataset.examples())
    {
        CHECK_EQ(model.predict_class(example.features), example.label);
        const auto p = model.predict_probabilities(example.features);
        CHECK_NEAR(p[0] + p[1], 1.0f, 1e-6f);
    }
}

TEST_CASE(Training, RejectsInvalidInputsWithoutChangingParameters)
{
    CHECK_THROW(ml::ShallowNetwork(0, 8, 2, 42), std::invalid_argument);
    CHECK_THROW(ml::ShallowNetwork(2, 0, 2, 42), std::invalid_argument);
    CHECK_THROW(ml::ShallowNetwork(2, 8, 1, 42), std::invalid_argument);
    ml::ShallowNetwork model(2, 8, 2, 42);
    const auto before = model.parameters();
    const auto stats = model.compute_training_statistics(data());
    CHECK_THROW(model.apply_gradient(stats.gradient_sum, 0, 0.1f), std::invalid_argument);
    CHECK_THROW(model.apply_gradient(stats.gradient_sum, 5, -0.1f), std::invalid_argument);
    auto gradient = stats.gradient_sum;
    gradient.back() = std::numeric_limits<float>::infinity();
    CHECK_THROW(model.apply_gradient(gradient, 5, 0.1f), std::invalid_argument);
    CHECK_EQ(before, model.parameters());
    CHECK_THROW(model.predict_class({1.0f}), std::invalid_argument);
    CHECK_THROW(model.compute_training_statistics(ml::Dataset::from_examples({{{1.0f, 2.0f}, 2}}, 2)), std::invalid_argument);
}
