#pragma once
#include "Light.h"
class DirectionLight :
	public Light
{
public:
	DirectionLight();
	DirectionLight(vec4 ambient, vec4 diffuse, vec4 specular,vec4 direction);
	DirectionLight(vec4 direction);
	void sendData(GLuint program, std::string name);

public:
	vec4 direction;
};

