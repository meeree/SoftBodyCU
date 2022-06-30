#version 450 core

out vec4 color;

layout (location=3) uniform mat4 pMat;
layout (location=4) uniform mat4 vMat;
layout (location=5) uniform mat4 mMat;
uniform int render_lines;
uniform float color_divisor;
uniform float color_off;
uniform float color_cutoff;
uniform vec3 campos;

in VS_OUT
{   
    vec3 position;
    vec3 normal;
    float voltage;
} fs_in;

vec3 gist_rainbow(float t)
{
    return clamp(abs(fract(t + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
}

vec3 red_to_green(float t)
{
    return mix(vec3(1, 0, 0), vec3(0, 1, 0), t);
}

void main(void)
{
    color = vec4(gist_rainbow(fs_in.voltage), 1.0);
    return;

    vec3 lightIntensities = vec3(0.4);
    vec3 ambient = vec3(0.6);

    vec3 nm = normalize(fs_in.normal);
    vec4 col = vec4(gist_rainbow(fs_in.voltage), 1.0);
    col = vec4(1.0);
    vec3 l = normalize(campos - fs_in.position);
    float brightness = max(dot(nm, l), 0.0);
    color = vec4((ambient + brightness * lightIntensities) * col.rgb, col.a);
    color.xyz = sqrt(color.xyz); // Gamma Correction.
}