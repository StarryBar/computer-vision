layout(location = 0) in vec3 velocity;
layout(location = 1) in vec3 color;
out vec4 vertexColor; 
out vec4 vertexPosition;
uniform float t;
uniform mat4 lookat;
uniform mat4 projection;
uniform int IS_INIT=0;
vec4 position=vec4(0.0,0.0,0.0,1.0);

void main()
{
    /*caculate the position*/
   // position.x =  0.0+velocity.x*t*0.001;
    //position.y =  0.1+-0.5*4.9*pow(10,-7)*t*t + velocity.y*t*0.001;
   // position.z =  0.0+velocity.z*t*0.001;

    float r = velocity.z+velocity.z*velocity.x*t*0.001;
    position.x =  r*cos(0.001*velocity.x*t);
    position.z =  0.0+r*sin(0.001*velocity.x*t);
    position.y =  0.0+velocity.y*t*0.0005;


    vertexColor = vec4(color, 1.0);
    vertexPosition = position;
    gl_Position = projection*lookat*(position);
}