layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 model_view;
uniform mat4 projection;

void main()
{
    gl_Position = projection*model_view*vec4(aPos,1.0);
}