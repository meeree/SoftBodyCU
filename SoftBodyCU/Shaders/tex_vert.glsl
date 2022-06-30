#version 450 core

layout (location=2) uniform mat4 pMat;
layout (location=3) uniform mat4 vMat;
layout (location=4) uniform mat4 mMat;

out VS_OUT
{   
    vec2 tex_coords;
} vs_out;

void main(void)
{
    switch(gl_VertexID)
    {
    case 0: vs_out.tex_coords = vec2(0.0, 0.0); break;
    case 1: vs_out.tex_coords = vec2(1.0, 0.0); break;
    case 2: vs_out.tex_coords = vec2(0.0, 1.0); break;
    case 3: vs_out.tex_coords = vec2(1.0, 0.0); break;
    case 4: vs_out.tex_coords = vec2(1.0, 1.0); break;
    default: case 5: vs_out.tex_coords = vec2(0.0, 1.0); break;
    }
    vec3 pos = 2.0*vec3(vs_out.tex_coords, 0.0) - vec3(1.0);
    gl_Position = pMat*vMat*mMat*vec4(pos, 1.0);
}
