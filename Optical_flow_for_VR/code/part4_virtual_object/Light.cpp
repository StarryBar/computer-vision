#include "Light.h"



Light::Light():
	ambient(vec4(0.0, 0.0, 0.0, 1.0)),
	diffuse (vec4(0.8, 0.8, 0.8, 1.0)),
	specular(vec4(0.2, 0.2, 0.2, 1.0))
{

}

Light::Light(vec4 ambient, vec4 diffuse, vec4 specular)
{
	this->ambient = ambient;
	this->diffuse = diffuse;
	this->specular = specular;
}

void Light::sendData(GLuint program, std::string name)
{
	setUniform(program, (name + ".ambient").c_str(), ambient);
	setUniform(program, (name + ".diffuse").c_str(), diffuse);
	setUniform(program, (name + ".specular").c_str(), specular);
}

