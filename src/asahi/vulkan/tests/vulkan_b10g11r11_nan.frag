/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#version 450
#extension GL_GOOGLE_include_directive : require
layout(location = 0) out vec4 o;
#include "vulkan_b10g11r11_nan_pattern.glsl"
void main() { o = vec4(pattern(uvec2(gl_FragCoord.xy)), 1.0); }
