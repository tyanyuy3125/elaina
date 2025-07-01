#pragma once
#include "core/common.h"

ELAINA_NAMESPACE_BEGIN

constexpr unsigned int MAX_TRAIN_DEPTH = 4;                         // the MC estimate collected before that depth is only used for training
constexpr unsigned int NETWORK_ALIGNMENT = 16;                      // both the cutlass and fully-fused network needs 16byte-aligned.
constexpr unsigned int MAX_RESOLUTION = 2048 * 2048;                // max size of the rendering frame
constexpr int TRAIN_BUFFER_SIZE = MAX_TRAIN_DEPTH * MAX_RESOLUTION; // [resolution-affected]
constexpr size_t TRAIN_BATCH_SIZE = 65'536 * 8;
constexpr size_t MIN_TRAIN_BATCH_SIZE = 65'536;   // the minimum batch size we can tolerate (to avoid unstable training)
constexpr int MAX_INFERENCE_NUM = MAX_RESOLUTION; // [resolution-affected]
constexpr float TRAIN_LOSS_SCALE = 128;
constexpr unsigned int LOSS_GRAPH_SIZE = 256;

namespace common2d
{
    constexpr unsigned int NUM_VMF_COMPONENTS = 8;            // fixed number of the vMF components
    constexpr unsigned int N_DIM_SPATIAL_INPUT = 2;           // how many dims do spatial data (i.e., positions, and optionally directions)
    constexpr unsigned int N_DIM_INPUT = N_DIM_SPATIAL_INPUT; // how many dims do the network input [spatial + auxiliary] have?
    constexpr unsigned int N_DIM_VMF = 4;                     // how many dims do a vMF component have?
    constexpr unsigned int N_DIM_OUTPUT = NUM_VMF_COMPONENTS * N_DIM_VMF + 1;
    constexpr unsigned int N_DIM_PADDED_OUTPUT = 48; // network output size with next_multiple of 16!
}

namespace common3d
{
    constexpr unsigned int NUM_VMF_COMPONENTS = 8;            // fixed number of the vMF components
    constexpr unsigned int N_DIM_SPATIAL_INPUT = 3;           // how many dims do spatial data (i.e., positions, and optionally directions)
    constexpr unsigned int N_DIM_INPUT = N_DIM_SPATIAL_INPUT; // how many dims do the network input [spatial + auxiliary] have?
    constexpr unsigned int N_DIM_VMF = 5;                     // how many dims do a vMF component have?
    constexpr unsigned int N_DIM_OUTPUT = NUM_VMF_COMPONENTS * N_DIM_VMF + 1;
    constexpr unsigned int N_DIM_PADDED_OUTPUT = 48; // network output size with next_multiple of 16!
}

ELAINA_NAMESPACE_END