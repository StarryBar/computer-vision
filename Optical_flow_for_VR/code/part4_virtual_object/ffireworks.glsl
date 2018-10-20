in vec4 vertexPosition;
in vec4 vertexColor;
out vec4 FragColor;

void main()
{
    if(vertexPosition.y<0.01)
        discard;
    FragColor = vertexColor;
}