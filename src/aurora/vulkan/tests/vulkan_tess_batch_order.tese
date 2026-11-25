/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
layout(triangles, equal_spacing, ccw) in;
layout(push_constant) uniform PC { uint draw, iter, spin_vs, spin_tcs, prev; } pc;
layout(set = 0, binding = 0, std430) buffer B
{
   uint vs_out[256], tcs_out[256], tes_out[256], tes_level[256];
} b;
layout(location = 0) patch in uint v;
void main()
{
   b.tes_out[pc.draw] = v;
   /* Equal spacing cuts each outer edge into `level` equal parts, so the
    * finest denominator on the edges is the level the stages used. */
   if (any(equal(gl_TessCoord, vec3(0)))) {
      uint level = 1;
      for (uint d = 1; d <= 4; d++) {
         const vec3 c = gl_TessCoord * float(d);
         if (all(lessThan(abs(c - round(c)), vec3(1e-4))))
            break;
         level = d + 1;
      }
      atomicMax(b.tes_level[pc.draw], level);
   }
   gl_Position = gl_TessCoord.x * gl_in[0].gl_Position +
                 gl_TessCoord.y * gl_in[1].gl_Position +
                 gl_TessCoord.z * gl_in[2].gl_Position;
}
