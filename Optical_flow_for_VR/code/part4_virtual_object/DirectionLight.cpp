#include "DirectionLight.h"



DirectionLight::DirectionLight():direction(vec4(0.1, 0.0, -1.0, 0.0))
{
}

DirectionLight::DirectionLight(vec4 ambient, vec4 diffuse, vec4 specular, vec4 direction):
	Light(ambient, diffuse, specular)
{
	this->direction = direction;
}

DirectionLight::DirectionLight(vec4 direction)
{
	this->direction = direction;
}



void DirectionLight::sendData( GLuint program, std::string name)
{
	Light::sendData(program, name);
	setUniform(program, (name + ".direction").c_str(), direction);
}

