#pragma once
#include <iostream>
#include<string>
#include<fstream>
#include<vector>
#include"Angel-yjc.h"


#define ImageWidth  64
#define ImageHeight 64
extern GLubyte Image[ImageHeight][ImageWidth][4];

#define	stripeImageWidth 32
extern GLubyte stripeImage[4 * stripeImageWidth];

//Read the data from the files and store to the vertices array
void readVertices(std::string fileName, std::vector<vec3> &vertices);

void readVertices(std::string fileName, std::vector<vec3> &vertices, std::vector<vec3> &normals);

void readVertices(std::string fileName, std::vector<vec3> &vertices, std::vector<vec3> &normals, std::vector<vec2> &textCoord);

void setRollingStatus(const vec4 &startPosition, const vec4 &currentTarget,
	vec4 &rollingDir, float &rollingDis, vec4 &rollingAxis);

void setUniform(GLuint program, std::string varname, const mat4 &value);
void setUniform(GLuint program, std::string varname, const GLint value);
void setUniform(GLuint program, std::string varname, const GLfloat value);
void setUniform(GLuint program, std::string varname, const vec4 value);
void setUniform(GLuint program, std::string varname, const vec3 value);

void image_set_up(void);