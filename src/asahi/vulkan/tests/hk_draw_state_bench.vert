#version 460
/* -DBASE: the position also reads gl_BaseInstance, so every draw republishes
 * its render state. */
layout(push_constant) uniform Push { vec4 offset; } push;
layout(location = 0) out vec3 color;
void main()
{
   vec2 p = vec2(gl_VertexIndex & 1, (gl_VertexIndex >> 1) & 1);
   color = vec3(p, 0.5) + push.offset.zzz;
#ifdef BASE
   p += vec2(gl_BaseInstance) * 0.01;
#endif
   gl_Position = vec4(p - 0.5 + push.offset.xy, 0.5, 1.0);
}
