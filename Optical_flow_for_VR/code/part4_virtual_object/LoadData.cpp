#include "LoadData.h"



void readVertices(std::string fileName, std::vector<vec3> &vertices)
{
	int numV;
	std::ifstream inFile(fileName, std::ios::in);
	if (!inFile)
	{
		std::cerr << "Unable to open file" << std::endl;
	}
	//delete the previous data and generate the new vertices array
	inFile >> numV;

	int dataSize = 0;
	while (!inFile.eof())
	{

		inFile >> dataSize;
		for (int j = 0; j < dataSize; j++)
		{	
			vec3 data;
			inFile >> data.x >> data.y >> data.z;
			vertices.push_back(data);
		}
	}
	inFile.close();
}

void readVertices(std::string fileName, std::vector<vec3>& vertices, std::vector<vec3>& normals)
{
	int numV;
	std::ifstream inFile(fileName, std::ios::in);
	if (!inFile)
	{
		std::cerr << "Unable to open file" << std::endl;
	}
	//delete the previous data and generate the new vertices array
	inFile >> numV;

	int dataSize = 0;
	while (!inFile.eof())
	{

		inFile >> dataSize;
			vec3 data1, data2, data3;
			inFile >> data1.x >> data1.y >> data1.z;
			vertices.push_back(data1);
			inFile >> data2.x >> data2.y >> data2.z;
			vertices.push_back(data2);
			inFile >> data3.x >> data3.y >> data3.z;
			vertices.push_back(data3);

			vec3 normal = cross((data2 - data1), (data3 - data2));
			normal = normalize(normal);
			normals.push_back(normal);
			normals.push_back(normal);
			normals.push_back(normal);
	}
	inFile.close();
}

void readVertices(std::string fileName, std::vector<vec3>& vertices, std::vector<vec3>& normals, std::vector<vec2>& textCoord)
{
	int numV;
	std::ifstream inFile(fileName, std::ios::in);
	if (!inFile)
	{
		std::cerr << "Unable to open file" << std::endl;
	}
	//delete the previous data and generate the new vertices array
	inFile >> numV;

	int dataSize = 0;
	while (!inFile.eof())
	{

		inFile >> dataSize;
		
		//Read the vertices position
		vec3 data1, data2, data3;
		vec2 data4;

		inFile >> data1.x >> data1.y >> data1.z;
		vertices.push_back(data1);
		inFile >> data4.x >> data4.y;
		textCoord.push_back(data4);

		inFile >> data2.x >> data2.y >> data2.z;
		vertices.push_back(data2);
		inFile >> data4.x >> data4.y;
		textCoord.push_back(data4);

		inFile >> data3.x >> data3.y >> data3.z;
		vertices.push_back(data3);
		inFile >> data4.x >> data4.y;
		textCoord.push_back(data4);


		//Caculate the normal vector
		vec3 normal = cross((data2 - data1), (data3 - data2));
		normal = normalize(normal);
		normals.push_back(normal);
		normals.push_back(normal);
		normals.push_back(normal);
	}
	inFile.close();
}

void setRollingStatus(const vec4 & startPosition, const vec4 & currentTarget, vec4 & rollingDir, float & rollingDis, vec4 & rollingAxis)
{
	rollingDir = normalize(currentTarget - startPosition);
	rollingDis = length(currentTarget - startPosition);
	rollingAxis = cross(rollingDir, vec4(0, -1, 0, 0));
}

void setUniform(GLuint program, std::string varname, const mat4 &value)
{
	glUseProgram(program);
	GLuint var_location;
	var_location = glGetUniformLocation(program, varname.c_str());
	glUniformMatrix4fv(var_location, 1, GL_TRUE, value);
}

void setUniform(GLuint program, std::string varname, const GLint value)
{
	glUseProgram(program);
	GLuint var_location;
	var_location = glGetUniformLocation(program, varname.c_str());
	glUniform1i(var_location, value);
}

void setUniform(GLuint program, std::string varname, const GLfloat value)
{
	glUseProgram(program);
	GLuint var_location;
	var_location = glGetUniformLocation(program, varname.c_str());
	glUniform1f(var_location, value);
}

void setUniform(GLuint program, std::string varname, const vec4 value)
{
	glUseProgram(program);
	GLuint var_location;
	var_location = glGetUniformLocation(program, varname.c_str());
	glUniform4fv(var_location, 1, value);
}

void setUniform(GLuint program, std::string varname, const vec3 value)
{
	glUseProgram(program);
	GLuint var_location;
	var_location = glGetUniformLocation(program, varname.c_str());
	glUniform3fv(var_location, 1, value);
}


GLubyte Image[ImageHeight][ImageWidth][4];
GLubyte stripeImage[4 * stripeImageWidth];

void image_set_up(void)
{
	int i, j, c;

	/* --- Generate checkerboard image to the image array ---*/
	for (i = 0; i < ImageHeight; i++)
		for (j = 0; j < ImageWidth; j++)
		{
			c = (((i & 0x8) == 0) ^ ((j & 0x8) == 0));

			if (c == 1) /* white */
			{
				c = 255;
				Image[i][j][0] = (GLubyte)c;
				Image[i][j][1] = (GLubyte)c;
				Image[i][j][2] = (GLubyte)c;
			}
			else  /* green */
			{
				Image[i][j][0] = (GLubyte)0;
				Image[i][j][1] = (GLubyte)150;
				Image[i][j][2] = (GLubyte)0;
			}

			Image[i][j][3] = (GLubyte)255;
		}

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	/*--- Generate 1D stripe image to array stripeImage[] ---*/
	for (j = 0; j < stripeImageWidth; j++) {
		/* When j <= 4, the color is (255, 0, 0),   i.e., red stripe/line.
		When j > 4,  the color is (255, 255, 0), i.e., yellow remaining texture
		*/
		stripeImage[4 * j] = (GLubyte)255;
		stripeImage[4 * j + 1] = (GLubyte)((j>4) ? 255 : 0);
		stripeImage[4 * j + 2] = (GLubyte)0;
		stripeImage[4 * j + 3] = (GLubyte)255;
	}

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	/*----------- End 1D stripe image ----------------*/

	/*--- texture mapping set-up is to be done in
	init() (set up texture objects),
	display() (activate the texture object to be used, etc.)
	and in shaders.
	---*/
}

void setLightingInfo(GLuint program, vec4 light_position ,mat4 lookat, vec4 ambient_product, vec4 diffuse_product, vec4 specular_product)
{
	//sending the shading info to shader
	setUniform(program, "ambient_product", ambient_product);
	setUniform(program, "diffuse_product", diffuse_product);
	setUniform(program, "specular_product", specular_product);
	setUniform(program, "light_pos_e", lookat*light_position);
}