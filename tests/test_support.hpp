#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Small dependency-free test runner. Failed checks stop the current test;
// the runner continues with the remaining registered cases.
namespace test
{
    using Function = void (*)();
    inline std::vector<std::pair<std::string, Function>> &cases()
    {
        static std::vector<std::pair<std::string, Function>> entries;
        return entries;
    }
    struct Register
    {
        Register(const char *name, Function function) { cases().emplace_back(name, function); }
    };
    class Check
    {
    public:
        Check(bool passed, const char *expression, const char *file, int line) : passed_(passed)
        {
            if (!passed_) message_ << file << ':' << line << ": " << expression << ' ';
        }
        template <typename T> Check &operator<<(const T &value)
        {
            if (!passed_) message_ << value;
            return *this;
        }
        ~Check() noexcept(false)
        {
            if (!passed_) throw std::runtime_error(message_.str());
        }
    private:
        bool passed_;
        std::ostringstream message_;
    };
    inline bool float_equal(float left, float right)
    {
        return left == right || std::abs(left - right) <= 4 * std::numeric_limits<float>::epsilon() *
            std::max({1.0f, std::abs(left), std::abs(right)});
    }
}

#define TEST_CASE(suite, name) \
    static void suite##_##name(); \
    static const test::Register register_##suite##_##name(#suite "." #name, suite##_##name); \
    static void suite##_##name()
#define CHECK_TRUE(expression) test::Check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
#define CHECK_FALSE(expression) CHECK_TRUE(!(expression))
#define CHECK_EQ(left, right) CHECK_TRUE((left) == (right))
#define CHECK_LT(left, right) CHECK_TRUE((left) < (right))
#define CHECK_LE(left, right) CHECK_TRUE((left) <= (right))
#define CHECK_NEAR(left, right, tolerance) CHECK_TRUE(std::abs((left) - (right)) <= (tolerance))
#define CHECK_FLOAT_EQ(left, right) CHECK_TRUE(test::float_equal((left), (right)))
#define CHECK_DOUBLE_EQ(left, right) CHECK_TRUE((left) == (right))
#define CHECK_THROW(statement, exception_type) \
    [&]() { bool caught = false; try { statement; } catch (const exception_type &) { caught = true; } \
        CHECK_TRUE(caught) << "Expected " #exception_type; }()
#define CHECK_ANY_THROW(statement) \
    [&]() { bool caught = false; try { statement; } catch (...) { caught = true; } CHECK_TRUE(caught); }()
#define CHECK_NO_THROW(statement) [&]() { statement; }()
