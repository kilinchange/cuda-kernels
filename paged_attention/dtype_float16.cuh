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

// FP32 accumulator vector types corresponding to Vec.
template <>
struct FloatVec<uint16_t> {
   using Type = float;
};
template <>
struct FloatVec<uint32_t> {
   using Type = float2;
};
template <>
struct FloatVec<uint2> {
   using Type = Float4_;
};
template <>
struct FloatVec<uint4> {
   using Type = Float8_;
};

// From float32 to float16.
inline __device__ void from_float(uint16_t& dst, float src) {
   dst = float_to_half(src);
}

inline __device__ void from_float(uint32_t& dst, float2 src) {
   dst = float2_to_half2(src);
}

inline __device__ void from_float(uint2& dst, Float4_ src) {
   dst.x = float2_to_half2(src.x);
   dst.y = float2_to_half2(src.y);
}

inline __device__ void from_float(uint4& dst, Float8_ src) {
   dst.x = float2_to_half2(src.x);
   dst.y = float2_to_half2(src.y);
   dst.z = float2_to_half2(src.z);
   dst.w = float2_to_half2(src.w);
}

// Vector fused multiply-add.
inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
   uint32_t d;
   asm volatile("v_pk_fma_f16 %0, %1, %2, %3;\n" : "=v"(d) : "v"(a), "v"(b), "v"(c));
   return d;
}

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
   return fma(h0_h0(a), b, c);
}

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
   uint2 d;
   d.x = fma(a.x, b.x, c.x);
   d.y = fma(a.y, b.y, c.y);
   return d;
}

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
   uint32_t s = h0_h0(a);
   uint2 d;
   d.x = fma(s, b.x, c.x);
   d.y = fma(s, b.y, c.y);
   return d;
}

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
   uint4 d;
   d.x = fma(a.x, b.x, c.x);
   d.y = fma(a.y, b.y, c.y);
   d.z = fma(a.z, b.z, c.z);
   d.w = fma(a.w, b.w, c.w);
   return d;
}

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
   uint32_t s = h0_h0(a);
   uint4 d;
   d.x = fma(s, b.x, c.x);
   d.y = fma(s, b.y, c.y);
   d.z = fma(s, b.z, c.z);
   d.w = fma(s, b.w, c.w);
   return d;
}

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
   float fa = half_to_float(a);
   float fb = half_to_float(b);
   return fa * fb + fc;
}

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
   float2 fa = half2_to_float2(a);
   float2 fb = half2_to_float2(b);
   return fma(fa, fb, fc);
}

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) {
   return fma(h0_h0(a), b, fc);
}

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
   Float4_ fd;
   fd.x = fma(a.x, b.x, fc.x);
   fd.y = fma(a.y, b.y, fc.y);
   return fd;
}

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
   uint32_t s = h0_h0(a);
   Float4_ fd;
   fd.x = fma(s, b.x, fc.x);
   fd.y = fma(s, b.y, fc.y);
   return fd;
}

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
   Float8_ fd;
   fd.x = fma(a.x, b.x, fc.x);
   fd.y = fma(a.y, b.y, fc.y);
   fd.z = fma(a.z, b.z, fc.z);
   fd.w = fma(a.w, b.w, fc.w);
   return fd;
}

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
   uint32_t s = h0_h0(a);
   Float8_ fd;
   fd.x = fma(s, b.x, fc.x);
   fd.y = fma(s, b.y, fc.y);
   fd.z = fma(s, b.z, fc.z);
   fd.w = fma(s, b.w, fc.w);
   return fd;
}
