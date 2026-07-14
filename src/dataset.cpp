#include "dataset.hpp"

#include <fstream>

// ------------ Dataset ------------

Dataset Dataset::load_csv(
    const std::string &path,
    std::size_t input_dimension)
{
    std::vector<Example> examples;

    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file: " + path);
    }
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
