#include "dataset.hpp"

#include <fstream>
#include <sstream>

// ------------ Dataset ------------

Dataset Dataset::load_csv(
    const std::string &path,
    std::size_t input_dimension)
{
    Dataset dataset({}, input_dimension);

    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open CSV file: " + path);
    }

    // Skip header line
    std::string header_line;
    if (!std::getline(file, header_line))
    {
        throw std::runtime_error("CSV file is empty: " + path);
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream line_stream(line);
        std::string cell;
        Example example;

        // Read features
        for (std::size_t i = 0; i < input_dimension; ++i)
        {
            if (!std::getline(line_stream, cell, ','))
            {
                throw std::runtime_error("Invalid CSV format: not enough features");
            }
            example.features.push_back(std::stof(cell));
        }

        // Read label
        if (!std::getline(line_stream, cell, ','))
        {
            throw std::runtime_error("Invalid CSV format: missing label");
        }
        example.label = static_cast<std::size_t>(std::stoul(cell));

        dataset.examples_.push_back(std::move(example));
    }

    return dataset;
}

Dataset::Dataset(std::vector<Example> examples, std::size_t input_dimension)
    : examples_(std::move(examples)), input_dimension_(input_dimension)
{
}

const std::vector<Example> &Dataset::examples() const noexcept
{
    return examples_;
}

std::size_t Dataset::size() const noexcept
{
    return examples_.size();
}

std::size_t Dataset::input_dimension() const noexcept
{
    return input_dimension_;
}
