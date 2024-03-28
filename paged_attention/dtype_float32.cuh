#pragma once

#include "attention_generic.cuh"

#include <stdint.h>

// FP32 vector types for Q, K, V.
template <>
struct Vec<float, 1> {
   using Type = float;
};
template <>
struct Vec<float, 2> {
   using Type = float2;
};
template <>
struct Vec<float, 4> {
   using Type = float4;
};
