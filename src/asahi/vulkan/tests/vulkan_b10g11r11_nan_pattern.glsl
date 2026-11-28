/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
// Test pattern of vulkan_b10g11r11_nan.c: exact B10G11R11 values, with NaN and +inf texels in every channel.
layout(push_constant) uniform PC { uint w; uint h; uint mode; uint nanbits; } pc;
vec3 pattern(uvec2 p)
{
   vec3 c = vec3(0.25 + float(p.x & 15u) / 64.0, 0.5, 0.75 + float(p.y & 15u) / 64.0);
   float nan = uintBitsToFloat(pc.nanbits), inf = uintBitsToFloat(0x7f800000u);
   uint a = p.x & 7u, b = p.y & 7u;
   if (a == 3u && b == 5u) c.x = nan;
   if (a == 1u && b == 1u) c.x = inf;
   if (a == 5u && b == 3u) c.y = nan;
   if (a == 7u && b == 0u) c.y = inf;
   if (a == 6u && b == 6u) c.z = nan;
   if (a == 2u && b == 7u) c.z = inf;
   return c;
}
