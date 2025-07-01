// #include "common.h"

// #include "util/vonmises.h"

// TEST_CASE("evalPoly", "[VonMises]")
// {
//     REQUIRE_THAT(evalPoly(1.14514f, COEF_LARGE[0], COEF_LARGE_ORDER), Catch::Matchers::WithinRel(0.4184690292340133f, 1e-5f));
//     REQUIRE_THAT_GPU(evalPoly(1.14514f, COEF_LARGE[0], COEF_LARGE_ORDER), Catch::Matchers::WithinRel(0.4184690292340133f, 1e-5f));
// }

// TEST_CASE("logModifiedBesselFn", "[VonMises]")
// {
//     const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
//     const float expected[] = {0.23591432f, 0.82399356f, 1.58530772f, 2.42497277f};
//     for (int i = 0; i < 4; ++i)
//     {
//         const auto input_item = input[i];
//         const auto expected_item = expected[i];
//         REQUIRE_THAT(logModifiedBesselFn(input_item), Catch::Matchers::WithinRel(expected_item, 1e-5f));
//         REQUIRE_THAT_GPU(logModifiedBesselFn(input_item), Catch::Matchers::WithinRel(expected_item, 1e-5f));
//     }
// }

// namespace
// {
//     float compute_mean(const std::vector<float> &samples)
//     {
//         float sum = std::accumulate(samples.begin(), samples.end(), 0.0);
//         return sum / samples.size();
//     }

//     float compute_circular_variance(const std::vector<float> &samples)
//     {
//         float cos_mean = 0.0f;
//         std::for_each(samples.begin(), samples.end(), [&](const auto sample) -> void
//                       { cos_mean += std::cos(sample); });
//         cos_mean /= samples.size();

//         float sin_mean = 0.0f;
//         std::for_each(samples.begin(), samples.end(), [&](const auto sample) -> void
//                       { sin_mean += std::sin(sample); });
//         sin_mean /= samples.size();

//         float R = std::sqrt(cos_mean * cos_mean + sin_mean * sin_mean);
//         return 1 - R;
//     }
// }

// TEST_CASE("VonMises", "[VonMises]")
// {
//     SECTION("log_prob, prob")
//     {
//         Allocator &alloc = *gpContext->alloc;
//         VonMises *vm_unified = alloc.new_object<VonMises>(4.2f);

//         VonMises &vm = *vm_unified;
//         const float input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
//         const float expected[] = {-6.18411160f, -2.16702533f, -0.23629522f, -2.16702533f, -6.18411160f};
//         const float expected_exp[] = {0.00206193f, 0.11451776f, 0.78954756f, 0.11451776f, 0.00206193f};
//         for (int i = 0; i < 5; ++i)
//         {
//             const auto input_item = input[i];
//             const auto expected_item = expected[i];
//             const auto expected_item_exp = expected_exp[i];
//             REQUIRE_THAT(vm.log_prob(input_item), Catch::Matchers::WithinRel(expected_item, 1e-5f));
//             REQUIRE_THAT_GPU(vm.log_prob(input_item), Catch::Matchers::WithinRel(expected_item, 1e-5f));
//             REQUIRE_THAT(vm.prob(input_item), Catch::Matchers::WithinRel(expected_item_exp, 1e-5f));
//             REQUIRE_THAT_GPU(vm.prob(input_item), Catch::Matchers::WithinRel(expected_item_exp, 1e-5f));
//         }
//     }

//     SECTION("sample - large kappa")
//     {
//         float kappa = 145.f;

//         PCGSampler *pcg_sampler = new PCGSampler();
//         pcg_sampler->setSeed(42);
//         pcg_sampler->initialize();
//         Sampler sampler(pcg_sampler);
//         VonMises vm(kappa);
//         const int num_samples = 10000;
//         std::vector<float> samples;

//         for (int i = 0; i < num_samples; ++i)
//         {
//             samples.push_back(vm.sample(sampler));
//         }

//         float sample_mean = compute_mean(samples);
//         REQUIRE(std::abs(sample_mean) < 0.1f);

//         float sample_variance = compute_circular_variance(samples);

//         float theoretical_variance = 1.0f - exp(logModifiedBesselFn(kappa, 1) - logModifiedBesselFn(kappa, 0));
//         REQUIRE(sample_variance == Catch::Approx(theoretical_variance).epsilon(0.05f));
//     }

//     SECTION("sample - small kappa")
//     {
//         float kappa = 1.45f;

//         PCGSampler *pcg_sampler = new PCGSampler();
//         pcg_sampler->setSeed(42);
//         pcg_sampler->initialize();
//         Sampler sampler(pcg_sampler);
//         VonMises vm(kappa);
//         const int num_samples = 1000000;
//         std::vector<float> samples;

//         for (int i = 0; i < num_samples; ++i)
//         {
//             samples.push_back(vm.sample(sampler));
//         }

//         float sample_mean = compute_mean(samples);
//         REQUIRE(std::abs(sample_mean) < 0.1f);

//         float sample_variance = compute_circular_variance(samples);

//         float theoretical_variance = 1.0f - exp(logModifiedBesselFn(kappa, 1) - logModifiedBesselFn(kappa, 0));
//         REQUIRE(sample_variance == Catch::Approx(theoretical_variance).epsilon(0.01f));
//     }

//     SECTION("d_log_prob_d_kappa - small")
//     {
//         float kappa = 1.45f;
//         float angle = 0.5f;

//         Allocator &alloc = *gpContext->alloc;
//         VonMises *vm_unified = alloc.new_object<VonMises>(kappa);

//         VonMises &vm = *vm_unified;
//         REQUIRE_THAT(vm.d_log_prob_d_kappa(angle), Catch::Matchers::WithinRel(0.29405486583709717f, 1e-5f));
//         REQUIRE_THAT_GPU(vm.d_log_prob_d_kappa(angle), Catch::Matchers::WithinRel(0.29405486583709717f, 1e-5f));
//     }

//     SECTION("d_log_prob_d_kappa - large")
//     {
//         float kappa = 14.5f;
//         float angle = 0.5f;

//         Allocator &alloc = *gpContext->alloc;
//         VonMises *vm_unified = alloc.new_object<VonMises>(kappa);

//         VonMises &vm = *vm_unified;
//         REQUIRE_THAT(vm.d_log_prob_d_kappa(angle), Catch::Matchers::WithinRel(-0.08729398250579834f, 1e-5f));
//         REQUIRE_THAT_GPU(vm.d_log_prob_d_kappa(angle), Catch::Matchers::WithinRel(-0.08729398250579834f, 1e-5f));
//     }
// }
