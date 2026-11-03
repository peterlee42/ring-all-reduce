#include "ml/dataset.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace
{
    std::vector<std::string> cells(const std::string &line)
    {
        std::vector<std::string> result;
        std::istringstream stream(line);
        std::string cell;
        while (std::getline(stream, cell, ',')) result.push_back(cell);
        if (!line.empty() && line.back() == ',') result.emplace_back();
        return result;
    }
}

ml::Dataset ml::Dataset::load_csv(const std::string &path, std::size_t input_dimension)
{
    if (input_dimension == 0) throw std::invalid_argument("Input dimension must be positive");
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Failed to open CSV file: " + path);
    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error("CSV file is empty: " + path);
    if (cells(line).size() != input_dimension + 1)
        throw std::runtime_error("CSV header must contain feature columns followed by one label column");
    std::vector<Example> examples;
    std::size_t row = 1;
    while (std::getline(file, line))
    {
        ++row;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto fields = cells(line);
        if (fields.size() != input_dimension + 1)
            throw std::runtime_error("Wrong column count on CSV row " + std::to_string(row));
        Example example;
        for (std::size_t i = 0; i < input_dimension; ++i)
        {
            std::size_t consumed = 0;
            const float value = std::stof(fields[i], &consumed);
            if (consumed != fields[i].size() || !std::isfinite(value))
                throw std::runtime_error("Invalid feature on CSV row " + std::to_string(row));
            example.features.push_back(value);
        }
        const auto &label = fields.back();
        const auto parsed = std::from_chars(label.data(), label.data() + label.size(), example.label);
        if (parsed.ec != std::errc{} || parsed.ptr != label.data() + label.size())
            throw std::runtime_error("Invalid label on CSV row " + std::to_string(row));
        examples.push_back(std::move(example));
    }
    if (file.bad()) throw std::runtime_error("Failed reading CSV file: " + path);
    if (examples.empty()) throw std::runtime_error("CSV contains no examples: " + path);
    return from_examples(std::move(examples), input_dimension);
}

ml::Dataset ml::Dataset::from_examples(std::vector<Example> examples, std::size_t input_dimension)
{
    if (input_dimension == 0) throw std::invalid_argument("Input dimension must be positive");
    for (const auto &example : examples)
    {
        if (example.features.size() != input_dimension)
            throw std::invalid_argument("Example feature dimension mismatch");
        for (const float value : example.features)
            if (!std::isfinite(value)) throw std::invalid_argument("Non-finite feature");
    }
    return Dataset(std::move(examples), input_dimension);
}

ml::Dataset ml::Dataset::shard(std::size_t rank, std::size_t workers) const
{
    if (workers == 0 || rank >= workers) throw std::invalid_argument("Invalid dataset shard");
    std::vector<Example> examples;
    for (std::size_t i = rank; i < examples_.size();)
    {
        examples.push_back(examples_[i]);
        if (workers >= examples_.size() - i) break;
        i += workers;
    }
    return Dataset(std::move(examples), input_dimension_);
}

ml::Dataset::Dataset(std::vector<Example> examples, std::size_t input_dimension)
    : examples_(std::move(examples)), input_dimension_(input_dimension) {}

const std::vector<ml::Example> &ml::Dataset::examples() const noexcept { return examples_; }
std::size_t ml::Dataset::size() const noexcept { return examples_.size(); }
std::size_t ml::Dataset::input_dimension() const noexcept { return input_dimension_; }
