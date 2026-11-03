#include "test_support.hpp"
#include <iostream>

int main()
{
    std::size_t failures = 0;
    for (const auto &[name, function] : test::cases())
    {
        try
        {
            function();
            std::cout << "PASS " << name << '\n';
        }
        catch (const std::exception &error)
        {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
        catch (...)
        {
            ++failures;
            std::cerr << "FAIL " << name << ": unknown exception\n";
        }
    }
    std::cout << test::cases().size() - failures << '/' << test::cases().size() << " passed\n";
    return failures == 0 ? 0 : 1;
}
