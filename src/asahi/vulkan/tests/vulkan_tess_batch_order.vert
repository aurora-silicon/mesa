/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(push_constant) uniform PC { uint draw, iter, spin_vs, spin_tcs, prev; } pc;
layout(set = 0, binding = 0, std430) buffer B
{
   uint vs_out[256], tcs_out[256], tes_out[256], tes_level[256];
} b;
void main()
{
   /* One invocation per draw writes, after spin_vs rounds of an LCG. */
   if (gl_VertexIndex == 0) {
      uint x = pc.draw * 0x9e3779b9u + pc.iter;
      for (uint i = 0; i < pc.spin_vs; i++)
         x = x * 1664525u + 1013904223u;
      b.vs_out[pc.draw] = x | 1u;
   }
   const vec2 tri[3] = vec2[](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0, 0.5));
   gl_Position = vec4(tri[gl_VertexIndex % 3], 0, 1);
}
