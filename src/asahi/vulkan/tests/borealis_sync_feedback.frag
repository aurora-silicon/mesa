/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Reads the attachment's own texel at a fixed offset, outside the render
 * area, and writes it back with the colour channels inverted.
 */
#version 450
layout(set = 0, binding = 0) uniform usampler2D src;
layout(push_constant) uniform Offset { ivec2 offset; } pc;
layout(location = 0) out uvec4 o;
void main()
{
   uvec4 t = texelFetch(src, ivec2(gl_FragCoord.xy) + pc.offset, 0);
   o = uvec4(t.rgb ^ uvec3(0xffu), t.a);
}
