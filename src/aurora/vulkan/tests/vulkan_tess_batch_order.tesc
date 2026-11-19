/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(vertices = 3) out;
layout(push_constant) uniform PC { uint draw, iter, spin_vs, spin_tcs, prev; } pc;
layout(set = 0, binding = 0, std430) buffer B
{
   uint vs_out[256], tcs_out[256], tes_out[256], tes_level[256];
} b;
layout(location = 0) patch out uint v;
void main()
{
   gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
   if (gl_InvocationID == 0) {
      /* What the vertex stage of this draw and the draws before it wrote:
       * all of them, or (prev) only the previous draw's. */
      uint x = 0;
      if (pc.prev != 0) {
         x = b.vs_out[pc.draw == 0 ? 0 : pc.draw - 1];
      } else {
         for (uint j = 0; j <= pc.draw; j++)
            x = x * 31u + b.vs_out[j];
      }
      uint s = 0x12345u ^ pc.draw;
      for (uint i = 0; i < pc.spin_tcs; i++)
         s = s * 1664525u + 1013904223u;
      x ^= s;
      b.tcs_out[pc.draw] = x;
      v = x;
      /* The tessellator stages size this draw's output from these. */
      const float level = float(1u + (x & 3u));
      gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = level;
      gl_TessLevelOuter[2] = gl_TessLevelInner[0] = level;
   }
}
