#version 450
/* A triangle covering the viewport at depth pc.z; scissors bound the draw. */
layout(push_constant) uniform Push { float z; } pc;
void main()
{
   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(p * 2.0 - 1.0, pc.z, 1.0);
}
