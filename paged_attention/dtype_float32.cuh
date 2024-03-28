#pragma once

#include "attention_generic.cuh"

#include <stdint.h>

// Define custom FP32 vector data types.
struct Float4_ {
   float2 x;
   float2 y;
};

struct Float8_ {
   float2 x;
   float2 y;
   float2 z;
   float2 w;
};

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

// FP32 accumulator vector types corresponding to Vec.
template <>
struct FloatVec<float> {
   using Type = float;
};
template <>
struct FloatVec<float2> {
   using Type = float2;
};
template <>
struct FloatVec<float4> {
   using Type = float4;
};

// From float to float.
inline __device__ void from_float(float& dst, float src) {
   dst = src;
}

inline __device__ void from_float(float2& dst, float2 src) {
   dst = src;
}

inline __device__ void from_float(float4& dst, float4 src) {
   dst = src;
}

// Vector fused multiply-add.
inline __device__ float fma(float a, float b, float c) {
   return a * b + c;
}

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
   float2 d;
   d.x = fma(a.x, b.x, c.x);
   d.y = fma(a.y, b.y, c.y);
   return d;
}

inline __device__ float2 fma(float a, float2 b, float2 c) {
   float2 d;
   d.x = fma(a, b.x, c.x);
   d.y = fma(a, b.y, c.y);
   return d;
}

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
   float4 d;
   d.x = fma(a.x, b.x, c.x);
   d.y = fma(a.y, b.y, c.y);
   d.z = fma(a.z, b.z, c.z);
   d.w = fma(a.w, b.w, c.w);
   return d;
}

inline __device__ float4 fma(float a, float4 b, float4 c) {
   float4 d;
   d.x = fma(a, b.x, c.x);
   d.y = fma(a, b.y, c.y);
   d.z = fma(a, b.z, c.z);
   d.w = fma(a, b.w, c.w);
   return d;
}

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
   Float4_ d;
   d.x = fma(a, b.x, c.x);
   d.y = fma(a, b.y, c.y);
   return d;
}

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
   Float8_ d;
   d.x = fma(a, b.x, c.x);
   d.y = fma(a, b.y, c.y);
   d.z = fma(a, b.z, c.z);
   d.w = fma(a, b.w, c.w);
   return d;
}
