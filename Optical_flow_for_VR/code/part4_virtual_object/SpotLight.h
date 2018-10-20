#pragma once
#include "Light.h"
class SpotLight :
	public Light
{
public:
	SpotLight();
	SpotLight(vec4 position,vec4 direction,vec3 attenuation,GLfloat e, GLfloat max_angle);
	///vec4 ambient, vec4 diffuse, vec4 specular, vec4 position, vec3 attenuation
	SpotLight(vec4 ambient, vec4 diffuse, vec4 specular, 
				vec4 position, vec4 direction, vec3 attenuation,
				GLfloat e, GLfloat max_angle);

	void sendData(GLuint program, std::string name);

public:
	vec4 position;
	vec4 direction;
	GLfloat max_angle_half;
	GLfloat e;
	vec3 attenuation;
};

