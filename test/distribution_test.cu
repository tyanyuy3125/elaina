// #include "common.h"

// #include "integrator/guided/distribution.h"

// using namespace common2d;

// namespace
// {
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

//     float compute_circular_mean(const std::vector<float> &samples)
//     {
//         float cos_sum = 0.0f;
//         std::for_each(samples.begin(), samples.end(), [&](const auto sample) -> void
//                       { cos_sum += std::cos(sample); });

//         float sin_sum = 0.0f;
//         std::for_each(samples.begin(), samples.end(), [&](const auto sample) -> void
//                       { sin_sum += std::sin(sample); });

//         return atan2(sin_sum, cos_sum);
//     }
// }

// TEST_CASE("VMKernel", "distribution")
// {
//     Allocator &alloc = *gpContext->alloc;
//     float *result = alloc.new_object<float>();
//     SECTION("eval")
//     {
//         VMKernel *vmk = alloc.new_object<VMKernel>(1.0f, 1.45f, M_PI_4);
//         GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {
//             *result = vmk->pdf(0.0f);
//         });
//         CUDA_SYNC_CHECK();
//         REQUIRE_THAT(*result, Catch::Matchers::WithinAbs(0.27751895785331726f, 1e-5f));

//         alloc.delete_object(vmk);
//     }

//     SECTION("eval - mean not normalized")
//     {
//         VMKernel *vmk = alloc.new_object<VMKernel>(1.0f, 1.45f, M_PI_4 + M_2PI);
//         GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {
//             *result = vmk->pdf(0.0f);
//         });
//         CUDA_SYNC_CHECK();
//         REQUIRE_THAT(*result, Catch::Matchers::WithinAbs(0.27751895785331726f, 1e-5f));

//         alloc.delete_object(vmk);
//     }

//     inter::vector<float, Allocator> samples(alloc);
//     samples.resize(1000000);

//     SECTION("sample")
//     {
//         std::vector<float> samples;
//         float gt_mean = M_PI_4;
//         VMKernel vmk(1.0f, 1.45f, gt_mean);
//         PCGSampler *pcg_sampler = new PCGSampler();
//         pcg_sampler->setSeed(42);
//         pcg_sampler->initialize();
//         Sampler sampler(pcg_sampler);
//         for (int i = 0; i < 1000000; ++i)
//         {
//             samples.push_back(vmk.sample(sampler));
//         }
//         float avg = compute_circular_mean(samples);
//         REQUIRE_THAT(avg, Catch::Matchers::WithinAbs(gt_mean, 1e-2) || Catch::Matchers::WithinAbs(gt_mean - M_PI, 1e-2));

//         float sample_variance = compute_circular_variance(samples);
//         float theoretical_variance = 1.0f - exp(logModifiedBesselFn(1.45f, 1) - logModifiedBesselFn(1.45f, 0));
//         REQUIRE(sample_variance == Catch::Approx(theoretical_variance).epsilon(0.01f));
//     }

//     SECTION("sample - mean not normalized")
//     {
//         std::vector<float> samples;
//         float gt_mean = M_PI_4;
//         VMKernel vmk(1.0f, 1.45f, gt_mean + M_2PI);
//         PCGSampler *pcg_sampler = new PCGSampler();
//         pcg_sampler->setSeed(42);
//         pcg_sampler->initialize();
//         Sampler sampler(pcg_sampler);
//         for (int i = 0; i < 1000000; ++i)
//         {
//             samples.push_back(vmk.sample(sampler));
//         }
//         float avg = compute_circular_mean(samples);
//         REQUIRE_THAT(avg, Catch::Matchers::WithinAbs(gt_mean, 1e-2) || Catch::Matchers::WithinAbs(gt_mean - M_PI, 1e-2));

//         float sample_variance = compute_circular_variance(samples);
//         float theoretical_variance = 1.0f - exp(logModifiedBesselFn(1.45f, 1) - logModifiedBesselFn(1.45f, 0));
//         REQUIRE(sample_variance == Catch::Approx(theoretical_variance).epsilon(0.01f));
//     }

//     SECTION("d_pdf_d_kappa, d_pdf_d_mean")
//     {
//         VMKernel *vmk = alloc.new_object<VMKernel>(1.0f, 1.45f, M_PI_4);
//         GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {
//             *result = vmk->d_pdf_d_kappa(0.0f);
//         });
//         CUDA_SYNC_CHECK();
//         REQUIRE_THAT(*result, Catch::Matchers::WithinAbs(0.034295544028282166f, 1e-5f));

//         GPUCall(ELAINA_DEVICE_LAMBDA_GLOBAL() {
//             *result = vmk->d_pdf_d_mean(0.0f);
//         });
//         CUDA_SYNC_CHECK();
//         REQUIRE_THAT(*result, Catch::Matchers::WithinAbs(-0.284541517496109f, 1e-5f));
//         alloc.delete_object(vmk);
//     }
//     alloc.delete_object(result);
// }

// TEST_CASE("VMM", "distribution")
// {
//     float data[] = {
//         0.0f,
//         0.0f,
//         0.0f};
//     VMM<1> vmm(data);
//     REQUIRE_THAT(vmm.pdf(0.0f), Catch::Matchers::WithinAbs(0.04624549299478531f, 1e-5f));

//     float data2[] = {
//         0.0f,
//         0.0f,
//         0.0f,
//         0.0f,
//         0.0f,
//         0.0f};
//     VMM<2> vmm2(data2);
//     REQUIRE_THAT(vmm2.pdf(0.0f), Catch::Matchers::WithinAbs(0.04624549299478531f, 1e-5f));
//     precision_t output[6];
//     REQUIRE_THAT(vmm2.gradients_probability(0.0f, output), Catch::Matchers::WithinAbs(0.04624549299478531f, 1e-5f));
//     REQUIRE_THAT(output[1], Catch::Matchers::WithinAbs(0.5f * -0.06688901782035828f, 1e-5f));
//     REQUIRE(output[0] == Catch::Approx(output[3]).epsilon(1e-3));
//     REQUIRE(output[1] == Catch::Approx(output[4]).epsilon(1e-3));
//     REQUIRE(output[2] == Catch::Approx(output[5]).epsilon(1e-3));
//     REQUIRE_THAT(output[2], Catch::Matchers::WithinAbs(0.5f * 4.042909562684827e-09f, 1e-5f));
//     REQUIRE_THAT(output[0], Catch::Matchers::WithinAbs(0.0f, 1e-5f));
// }

// TEST_CASE("VMM 2", "distribution")
// {
//     float data[] = {
//         -0.3391095697879791f, 1.3653955459594727f, -0.11165934801101685f,
//         0.7329881191253662f, 1.1205719709396362f, -1.145609736442566f,
//         1.5198860168457031f, -0.962236225605011f, 1.4103161096572876f};
//     VMM<3> vmm(data);
//     precision_t output[9];
//     REQUIRE_THAT(vmm.gradients_probability(0.0f, output), Catch::Matchers::WithinAbs(0.11850630f, 1e-5f));
//     float expected_output[] = {
//         -0.016046222299337387f, -5.7009561714949086e-05f, -2.110011519107502e-05f,
//         -0.011129779741168022f, -0.007846416905522346f, -0.031608663499355316f,
//         0.00756735447794199f, 0.015586040914058685f, 0.0389787033200264f};
//     for (int i = 0; i < 9; ++i)
//     {
//         REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected_output[i], 1e-5f));
//     }
// }
