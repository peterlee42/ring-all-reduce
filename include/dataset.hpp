#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct Example
{
    std::vector<float> features;
    float label;
};

class Dataset
{
public:
    static Dataset load_csv(
        const std::string &path,
        std::size_t input_dimension);

    const std::vector<Example> &examples() const noexcept;
    std::size_t size() const noexcept;
    std::size_t input_dimension() const noexcept;

private:
    Dataset(std::vector<Example> examples, std::size_t input_dimension);

    std::vector<Example> examples_;
    std::size_t input_dimension_;
};
