#pragma once

#include <stdint.h>

// A vector type to store Q, K, V elements.
template <typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
template <typename T>
struct FloatVec {};
