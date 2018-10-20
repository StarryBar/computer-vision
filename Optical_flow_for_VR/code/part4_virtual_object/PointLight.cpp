#include "PointLight.h"



PointLight::PointLight()
{
	
}

PointLight::PointLight(vec4 position, vec3 attenuation)
{	
	this->position = position;
	this->attenuation = attenuation;
}

PointLight::PointLight(vec4 ambient, vec4 diffuse, vec4 specular, vec4 position, vec3 attenuation):
		Light(ambient, diffuse, specular)
{
	this->position = position;
	this->attenuation = attenuation;
}

void PointLight::sendData(GLuint program, std::string name)
{
	Light::sendData(program, name);
	setUniform(program, (name + ".position").c_str(), position);
	setUniform(program, (name + ".attenuation").c_str(), attenuation);
}

