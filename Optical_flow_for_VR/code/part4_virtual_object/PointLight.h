#pragma once
#include "Light.h"
class PointLight :
	public Light
{
public:
	PointLight();
	PointLight(vec4 position, vec3 attenuation);
	///vec4 ambient, vec4 diffuse, vec4 specular, vec4 position, vec3 attenuation
	PointLight(vec4 ambient, vec4 diffuse, vec4 specular, vec4 position, vec3 attenuation);
	
	void sendData(GLuint program, std::string name);


public:
	vec4 position;
	vec3 attenuation; //[const, linear, quad]

};

