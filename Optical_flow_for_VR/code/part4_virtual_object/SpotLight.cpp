#include "SpotLight.h"



SpotLight::SpotLight()
{
}

SpotLight::SpotLight(vec4 position, vec4 direction, vec3 attenuation, GLfloat e, GLfloat max_angle)
{
	this->position = position;
	this->direction = direction;
	this->attenuation = attenuation;
	this->e = e;
	this->max_angle_half = max_angle;
}

SpotLight::SpotLight(vec4 ambient, vec4 diffuse, vec4 specular, 
	vec4 position, vec4 direction, vec3 attenuation, 
	GLfloat e, GLfloat max_angle):
	Light(ambient,diffuse,specular)
{
	this->position = position;
	this->direction = direction;
	this->attenuation = attenuation;
	this->e = e;
	this->max_angle_half = max_angle;
}

void SpotLight::sendData(GLuint program, std::string name)
{
	Light::sendData(program, name);
	setUniform(program, (name + ".direction").c_str(), direction);
	setUniform(program, (name + ".position").c_str(), position);
	setUniform(program, (name + ".attenuation").c_str(), attenuation);
	setUniform(program, (name + ".e").c_str(), e);
	setUniform(program, (name + ".max_angle_half").c_str(), max_angle_half);
}
