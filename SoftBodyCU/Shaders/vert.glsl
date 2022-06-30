#version 450 core

layout (location=0) in vec3 position;
layout (location=1) in float voltage;
layout (location=2) in vec3 normal;

layout (location=3) uniform mat4 pMat;
layout (location=4) uniform mat4 vMat;
layout (location=5) uniform mat4 mMat;
uniform int render_lines;

out VS_OUT
{   
    vec3 position;
    vec3 normal;
    float voltage;
} vs_out;

void main(void)
{
    gl_Position = pMat * vMat * mMat * vec4(position, 1.0);
    vs_out.position = vec3(mMat * vec4(position, 1.0)); 
    vs_out.normal = mat3(transpose(inverse(mMat))) * normal;
    vs_out.voltage = voltage;
}
