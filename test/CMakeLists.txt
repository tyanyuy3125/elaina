add_executable(tests
common.h

test_main.cpp
vonmises_test.cu
distribution_test.cu
beta_test.cu
)

target_link_libraries(tests PRIVATE Catch2::Catch2 elaina)
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES "75")

include(CTest)
enable_testing()
