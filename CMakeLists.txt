cmake_minimum_required(VERSION 3.19)
project(elaina
    DESCRIPTION "Elaina is a wavefront implementation of Monte-Carlo PDE solver."
    LANGUAGES C CXX CUDA
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###############################################################################
# C++ compiler setup
###############################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions ("$<$<CONFIG:Debug>:ELAINA_DEBUG_BUILD>")

###############################################################################
# CUDA compiler setup
###############################################################################
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress=20012,20014")
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_VERBOSE_BUILD ON)

if (MSVC)
    add_compile_options(/wd4819)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()
find_library(
	CUDA_CUBLASLT_LIBRARY cublasLt
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)

# Try to detect architecture of all GPUs in the system
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(DETECTED_CUDA_ARCHS)
string(STRIP "${DETECTED_CUDA_ARCHS}" DETECTED_CUDA_ARCHS)
string(REPLACE " " ";" DETECTED_CUDA_ARCHS "${DETECTED_CUDA_ARCHS}")
message(STATUS "Raw detected architectures: ${DETECTED_CUDA_ARCHS}")
if(DETECTED_CUDA_ARCHS)
    # Convert detected architecture format (remove decimal points)
    foreach(arch ${DETECTED_CUDA_ARCHS})
        string(REGEX REPLACE "\\." "" arch_num ${arch})
        list(APPEND CUDA_ARCHS_PROCESSED ${arch_num})
    endforeach()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS_PROCESSED})
else()
    # Fallback to common modern architectures
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89")
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (NOT CUDA_INCLUDE_DIRS)
	message(STATUS "Setting CUDA include directory to: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
	set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

add_library (elaina_cuda_cfg INTERFACE)
target_compile_options (
	elaina_cuda_cfg
	INTERFACE
		"$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler>"
		# Do not add debugging symbols when generating PTX files for OptiX (not supported).
		"$<$<COMPILE_LANGUAGE:CUDA>:$<IF:$<AND:$<CONFIG:Debug>,$<NOT:$<BOOL:$<TARGET_PROPERTY:CUDA_PTX_COMPILATION>>>>,-G;-g,-lineinfo>>"
)

###############################################################################
# Dependencies and project setup
###############################################################################

add_subdirectory(ext)
add_subdirectory(core/math)

###############################################################################
# include files
###############################################################################

set(ELAINA_INCLUDE_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
    # from ./ext
    # ${ASSIMP_INCLUDE_DIRS}
    ${IMAGEIO_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
    ${TCNN_INCLUDE_DIRS}
    # from ./core/math
    ${KRR_MATH_INCLUDE_DIRS}
    # cuda
    ${CUDA_INCLUDE_DIRS}
    # indicators
    ${INDICATORS_INCLUDE_DIRS}
    # lbvh
    ${LBVH_INCLUDE_DIRS}
    # magic_enum
    ${MAGIC_ENUM_INCLUDE_DIRS}
    # openvdb
    ${OPENVDB_INCLUDE_DIRS}
    ${NANOVDB_INCLUDE_DIRS}
)

###############################################################################
# target setting
###############################################################################
set(SOURCES main.cpp exec.cu)

set(ELAINA_CORE_HEADER
    core/common.h
    core/config.h
    core/file.h
    core/interop.h
    core/logger.h
    core/evaluation_grid.h
    core/sampler.h
    core/problem.h
    core/taggedptr.h
    core/device/cuda.h
    core/device/context.h
    core/device/memory.h

    core/texture.h
    core/device/buffer.h
)

set(ELAINA_CORE_SOURCE
    core/file.cpp
    core/interop.cpp
    core/logger.cpp
    core/problem.cu
    core/device/context.cpp
    
    core/texture.cu
)

set(ELAINA_UTIL_HEADER
    util/check.h
    util/ema.h
    util/film.h
    util/hash.h
    util/image.h
    util/math_utils.h
    util/sampling.h
    util/transformation.h
    util/vmf.h
    util/vonmises.h
    util/network.h
    util/tonemapping.cuh
    util/green.h
    util/convert.h
)

set(ELAINA_UTIL_SOURCE
    util/image.cpp
)

set(ELAINA_INTEGRATOR_GUIDED_HEADER
    integrator/guided/integrator.h
    integrator/guided/parameters.h
    integrator/guided/distribution.h
    integrator/guided/train.h
    integrator/guided/guideditem.h
    integrator/guided/guided.h
)

set(ELAINA_INTEGRATOR_UNIFORM_HEADER
    integrator/uniform/integrator.h
    integrator/uniform/workitem.h
    integrator/uniform/workqueue.h
)

set(ELAINA_INTEGRATOR_HEADER
    ${ELAINA_INTEGRATOR_GUIDED_HEADER}
    ${ELAINA_INTEGRATOR_UNIFORM_HEADER}
    integrator/common.h
)

set(ELAINA_INTEGRATOR_GUIDED_SOURCE
    integrator/guided/integrator.cu
)

set(ELAINA_INTEGRATOR_UNIFORM_SOURCE
    integrator/uniform/integrator.cu
)

set(ELAINA_INTEGRATOR_SOURCE
    ${ELAINA_INTEGRATOR_GUIDED_SOURCE}
    ${ELAINA_INTEGRATOR_UNIFORM_SOURCE}
)

###############################################################################
# automatically creating definitions of structure of arrays (soa)
###############################################################################
add_executable(soac util/soac.cpp)
add_executable (elaina::soac ALIAS soac)

target_compile_options(soac PUBLIC ${CMAKE_CXX_FLAGS})
set_target_properties (soac PROPERTIES OUTPUT_NAME soac)

set(SOA_INCLUDE_GUIDED_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/integrator/guided")
set(SOA_INCLUDE_UNIFORM_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/integrator/uniform")

file(MAKE_DIRECTORY "${SOA_INCLUDE_GUIDED_DIR}")
file(MAKE_DIRECTORY "${SOA_INCLUDE_UNIFORM_DIR}")

# workitem
# 2d
add_custom_command (OUTPUT ${SOA_INCLUDE_UNIFORM_DIR}/uniform2d_workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/uniform/uniform2d_workitem.soa > ${SOA_INCLUDE_UNIFORM_DIR}/uniform2d_workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/uniform/uniform2d_workitem.soa)
set (ELAINA_WORKITEM2D_SOA_GENERATED 
	${SOA_INCLUDE_UNIFORM_DIR}/uniform2d_workitem_soa.h
)
add_custom_target (elaina_workitem2d_soa_generated DEPENDS ${ELAINA_WORKITEM2D_SOA_GENERATED})

# 3d
add_custom_command (OUTPUT ${SOA_INCLUDE_UNIFORM_DIR}/uniform3d_workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/uniform/uniform3d_workitem.soa > ${SOA_INCLUDE_UNIFORM_DIR}/uniform3d_workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/uniform/uniform3d_workitem.soa)
set (ELAINA_WORKITEM3D_SOA_GENERATED 
	${SOA_INCLUDE_UNIFORM_DIR}/uniform3d_workitem_soa.h
)
add_custom_target (elaina_workitem3d_soa_generated DEPENDS ${ELAINA_WORKITEM3D_SOA_GENERATED})

# guideditem
# 2d
add_custom_command (OUTPUT ${SOA_INCLUDE_GUIDED_DIR}/guided2d_workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/guided/guided2d_workitem.soa > ${SOA_INCLUDE_GUIDED_DIR}/guided2d_workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/guided/guided2d_workitem.soa)
set (ELAINA_GUIDED2D_SOA_GENERATED 
	${SOA_INCLUDE_GUIDED_DIR}/guided2d_workitem_soa.h
)
add_custom_target (elaina_guided2d_soa_generated DEPENDS ${ELAINA_GUIDED2D_SOA_GENERATED})

# 3d
add_custom_command (OUTPUT ${SOA_INCLUDE_GUIDED_DIR}/guided3d_workitem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/guided/guided3d_workitem.soa > ${SOA_INCLUDE_GUIDED_DIR}/guided3d_workitem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/integrator/guided/guided3d_workitem.soa)
set (ELAINA_GUIDED3D_SOA_GENERATED 
	${SOA_INCLUDE_GUIDED_DIR}/guided3d_workitem_soa.h
)
add_custom_target (elaina_guided3d_soa_generated DEPENDS ${ELAINA_GUIDED3D_SOA_GENERATED})

###############################################################################
# linking and executables
###############################################################################
set(ELAINA_LIBS_ALL
    # assimp
    cuda
    cublas
    # zlibstatic
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    krr_math
    tiny-cuda-nn
    elaina_cuda_cfg
    # openvdb
    openvdb
    nanovdb
)

add_library(elaina 
${ELAINA_CORE_HEADER}
${ELAINA_CORE_SOURCE}
${ELAINA_UTIL_HEADER}
${ELAINA_UTIL_SOURCE}
${ELAINA_INTEGRATOR_HEADER}
${ELAINA_INTEGRATOR_SOURCE}
${ELAINA_SOA_GENERATED}
)

target_include_directories(elaina PUBLIC ${ELAINA_INCLUDE_ALL})
add_dependencies(elaina elaina_workitem2d_soa_generated)
add_dependencies(elaina elaina_workitem3d_soa_generated)
add_dependencies(elaina elaina_guided2d_soa_generated)
add_dependencies(elaina elaina_guided3d_soa_generated)
target_link_libraries(elaina PUBLIC ${ELAINA_LIBS_ALL})
target_link_libraries(elaina PRIVATE ${ELAINA_LIBS_ALL_PRIVATE})
set_target_properties(elaina PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_compile_definitions(elaina PUBLIC TCNN_MIN_GPU_ARCH=${CMAKE_CUDA_ARCHITECTURES})
target_compile_definitions(elaina PUBLIC -DFCPW_USE_GPU)

add_executable(elaina-exec ${SOURCES})
target_link_libraries(elaina-exec PUBLIC elaina)
set_target_properties(elaina-exec PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

###############################################################################
# asset
###############################################################################

set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
set(ASSETS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/data)

add_custom_command(
    TARGET elaina-exec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${ASSETS_DIR} ${ASSETS_OUTPUT_DIR}
)

###############################################################################
# unit test integration
###############################################################################

option(ELAINA_ENABLE_TESTS "Enable compilation of tests" OFF)
set(ELAINA_ENABLE_TESTS ON)
if (ELAINA_ENABLE_TESTS)
    add_subdirectory(test)
endif ()
