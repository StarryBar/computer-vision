layout(location = 0) in  vec3 vPosition;
layout(location = 1) in  vec3 vcolor;

out vec4 color;


uniform mat4 model_view;
uniform mat4 projection;

void main()
{
    color = vec4(vcolor,1.0);
    gl_Position = projection*model_view*vec4(vPosition,1.0);
}