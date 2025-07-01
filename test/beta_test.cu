#include "common.h"
#include "util/beta.h"
#include <cmath>
#include <limits>

TEST_CASE("eval", "[Beta]") {
    // Test case 1: alpha = 1, beta = 1 (Uniform distribution)
    {
        BetaDistribution beta(1.0f, 1.0f);
        REQUIRE(beta.eval(0.0f) == Catch::Approx(1.0f)); // x = 0
        REQUIRE(beta.eval(0.5f) == Catch::Approx(1.0f)); // x = 0.5
        REQUIRE(beta.eval(1.0f) == Catch::Approx(1.0f)); // x = 1
    }

    // Test case 2: alpha = 2, beta = 2 (Symmetric distribution)
    {
        BetaDistribution beta(2.0f, 2.0f);
        REQUIRE(beta.eval(0.0f) == Catch::Approx(0.0f)); // x = 0
        REQUIRE(beta.eval(0.5f) == Catch::Approx(1.5f)); // x = 0.5
        REQUIRE(beta.eval(1.0f) == Catch::Approx(0.0f)); // x = 1
    }

    // Test case 3: alpha = 0.5, beta = 0.5 (U-shaped distribution)
    {
        BetaDistribution beta(0.5f, 0.5f);
        REQUIRE(beta.eval(0.0f) == Catch::Approx(std::numeric_limits<float>::infinity())); // x = 0
        REQUIRE(beta.eval(0.5f) == Catch::Approx(0.6366197723675814f)); // x = 0.5
        REQUIRE(beta.eval(1.0f) == Catch::Approx(std::numeric_limits<float>::infinity())); // x = 1
    }

    // Test case 4: alpha = 2, beta = 5 (Skewed distribution)
    {
        BetaDistribution beta(2.0f, 5.0f);
        REQUIRE(beta.eval(0.0f) == Catch::Approx(0.0f)); // x = 0
        REQUIRE(beta.eval(0.2f) == Catch::Approx(2.4576f)); // x = 0.2
        REQUIRE(beta.eval(1.0f) == Catch::Approx(0.0f)); // x = 1
    }

    // Test case 5: alpha = 5, beta = 2 (Skewed distribution)
    {
        BetaDistribution beta(5.0f, 2.0f);
        REQUIRE(beta.eval(0.0f) == Catch::Approx(0.0f)); // x = 0
        REQUIRE(beta.eval(0.8f) == Catch::Approx(2.4576f)); // x = 0.8
        REQUIRE(beta.eval(1.0f) == Catch::Approx(0.0f)); // x = 1
    }
}
