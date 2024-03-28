#pragma once

#include "attention_generic.cuh"

#include <stdint.h>

// FP16 vector types for Q, K, V.
template <>
struct Vec<uint16_t, 1> {
   using Type = uint16_t;
};
template <>
struct Vec<uint16_t, 2> {
   using Type = uint32_t;
};
template <>
struct Vec<uint16_t, 4> {
   using Type = uint2;
};
template <>
struct Vec<uint16_t, 8> {
   using Type = uint4;
};
