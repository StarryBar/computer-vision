#pragma once
#include"mat-yjc-new.h"
#include "LoadData.h"
class Light
{
public:
	///ambient, diffuse, specular
	///R G B
	Light();
	Light(vec4 ambient, vec4 diffuse, vec4 specular);
	virtual void sendData(GLuint program, std::string name);
public:
	vec4 ambient;
	vec4 diffuse;
	vec4 specular;
};

