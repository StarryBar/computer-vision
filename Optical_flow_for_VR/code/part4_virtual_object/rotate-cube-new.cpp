#include "Angel-yjc.h"
#include"LoadData.h"
#include "rotate-cube-new.h"
#include "menu.h"
#include"Light.h"
#include"DirectionLight.h"
#include"PointLight.h"
#include"SpotLight.h"
#include "CommandCode.h"
#include "ReadFromOpticalFlow.h"

std::vector<std::vector<int>> commandList;
typedef Angel::vec3  color3;
typedef Angel::vec3  point3;
std::ifstream inFile;
GLuint Angel::InitShader(const char* vShaderFile, const char* fShaderFile);

GLuint program;			/* shader program  that has ligth*/
GLuint noLightProgram;	/* shader program that has no light*/
GLuint fireWorksProgram;

GLuint axis_buffer;	  /*vertex buffer object id for axiss*/
GLuint floor_buffer;  /* vertex buffer object id for floor */
GLuint sphere_buffer; /*vertex buffer object id for sphere*/
GLuint fireworks_buffer;

// Projection transformation parameters
GLfloat  fovy = 60.0;  // Field-of-view in Y direction angle (in degrees)
GLfloat  aspect;       // Viewport aspect ratio
GLfloat  zNear = 1.0, zFar = 20.0;

//The animation parameters
int speed=1;
GLfloat t = 0;
GLfloat angle = 0.0; // rotation angle in degrees
GLfloat rollingDis = 0.0, currentDis = 0.0; //the rolling distance and current distance
vec4 rollingDir, rollingAxis;
const float d = 1.6*3.15 / 360;
vec4 posA = vec4(3.0, 1.0, 5.0, 1.0);
vec4 posB = vec4(-1.0, 1.0, -4.0, 1.0);
vec4 posC = vec4(3.5, 1, -2.5, 1.0);
vec4 currentTarget, startPostion;
mat4 accumRota;
//The clock of the fireworks
struct Fireworks_Clock
{
	GLfloat run_time_elapse;	//will just keep increasing
	GLint	loop_num;				//to comupte the loop
	GLfloat animate_length;		//How long the animate persisit
	GLfloat clock_start_point;	//the time of the offset, to record the begin time of the clock
	GLfloat clock;				//clock = run_time_elapse - clock_start_point
	GLfloat fireworks_time;	   //the actual time used in firework. 
									//fireworks_time = clock - loop_num*animate_length
};

Fireworks_Clock clock;

vec4 init_eye(7.0, 3.0, -10.0, 1.0); // initial viewer position
vec4 eye = init_eye;               // current viewer position

int beginRollingFlag = 1;//1: animation; 0: non-animation. Toggled by right_click or 'A'
int animationFlag = 0; // 1:beginRolling; 0:stop-rollinng  . Toggled by 'b' or 'B'


//the ligth source data
vec3 light_position = vec3(-14.0, 12.0, -3.0);
GLfloat shininess = 125.0;
DirectionLight dir;
DirectionLight null_dir(
		vec4(0.0, 0.0, 0.0, 0.0),  //ambient
		vec4(0.0, 0.0, 0.0, 0.0),  //diffuse
		vec4(0.0, 0.0, 0.0, 0.0),  //specular
		vec4(0.0, 0.0, 0.0, 0.0) //direction
);

PointLight point_light(
	vec4(0.0, 0.0, 0.0, 1.0),  //ambient
	vec4(1.0, 1.0, 1.0, 1.0),  //diffuse
	vec4(1.0, 1.0, 1.0, 1.0),  //specular
	vec4(-14.0, 12.0, -3.0, 1.0), //world position
	vec3(2.00, 0.01, 0.001)  //attenuation: const, linear, quad 
);

// a empty point light, has no effect
PointLight null_point(
	vec4(0.0, 0.0, 0.0, 0.0),  //ambient
	vec4(0.0, 0.0, 0.0, 0.0),  //diffuse
	vec4(0.0, 0.0, 0.0, 0.0),  //specular
	vec4(0.0, 0.0, 0.0, 0.0), //world position
	vec3(1.0, 1.0, 1.0)  //attenuation: const, linear, quad 
);

//Spot light
SpotLight spot_light(
	vec4(0.0, 0.0, 0.0, 1.0),  //ambient
	vec4(1.0, 1.0, 1.0, 1.0),  //diffuse
	vec4(1.0, 1.0, 1.0, 1.0),  //specular
	vec4(-14.0, 12.0, -3.0, 1.0), //world position
	vec4(-6.0, 0.0, -4.5, 1.0), //world target
	vec3(2.00, 0.01, 0.001),  //attenuation:
	1.50,					  //exponent	
	20				  //curoff angle
);

//a empty light, has no effect
SpotLight null_spot(
	vec4(0.0, 0.0, 0.0, 0.0),
	vec4(0.0, 0.0, 0.0, 0.0),
	vec4(0.0, 0.0, 0.0, 0.0),
	vec4(0.0, 0.0, 0.0, 0.0),
	vec4(0.0, 0.0, 0.0, 0.0),
	vec3(1.0, 1.0, 1.0),
	0.0,
	0.0
);

mat4 shadow_matrix = mat4(
	light_position.y,	0.0,		0.0,				0.0,
	-light_position.x,	0.0, -light_position.z,		   -1.0,
	0.0,				0.0, light_position.y,			0.0,
	0.0,				0.0,		0.0,			light_position.y
);

//the sphere data

#define VERTICAL			
#define EYE					000
#define WORLD				001
#define SPHERE_TEXTURE_MODE 
std::vector<point3> sphere_point; 
std::vector<color3> sphere_color; //yellow
std::vector<vec3>	sphere_normal;
	//the sphere lighting color data
	vec4 sphere_ambient = vec4(0.2, 0.2, 0.2, 1.0);
	vec4 sphere_diffuse = vec4(1.0, 0.84, 0.0, 1.0);
	vec4 sphere_specular = vec4(1.0, 0.84, 0.0, 1.0);
	GLfloat sphere_shininess = 125.0;

//the floor data
std::vector<point3> floor_point; // positions for all vertices
std::vector<color3> floor_color;  // green
std::vector<vec3>	floor_normal;
std::vector<vec2>   floor_textCoord;
	//the floor lighting color data
	vec4 floor_ambient = vec4(0.2, 0.2, 0.2, 0.2);
	vec4 floor_diffuse = vec4(0.0, 1.0, 0.0, 1.0);
	vec4 floor_specular = vec4(0.0, 0.0, 0.0, 1.0);
	GLfloat floor_shininess = 125.0;

std::vector<vec3> velocitys(300);
std::vector<vec3> colors(300);

//the axis data
const int axis_NumVertices = 6;
point3 axis_points[6] = {
	point3(0.0, 0.0, 0.0),   //the x axis
	point3(10.0,  0.0,  0.0),

	point3(0.0, 0.0, 0.0),   //the y axis
	point3(0.0, 10.0,  0.0),

	point3(0.0, 0.0, 0.0),  //the z axis
	point3(0.0, 0.0, 10.0)
};
color3 axis_color[6] = {
	color3(1.0, 0.0, 0.0),  // the x axis red
	color3(1.0, 0.0, 0.0),  // the x axis red

	color3(1.0, 0.0, 1.0),  // the y axis magenta
	color3(1.0, 0.0, 1.0),  // the y axis magenta

	color3(0.0, 0.0, 1.0),  // the z axis blue
	color3(0.0, 0.0, 1.0),  // the z axis blue
};

/*------------Texture Data--------------*/
GLuint texture_2D;
GLuint texture_1D;


//----------------------------------------------------------------------------
int Index = 0;

void sendingData()
{
	//Senting data to Sphere buffer
	readVertices("Sphere1024.txt", sphere_point,sphere_normal);
	sphere_color.assign(sphere_point.size(), color3(1.0, 0.84, 0.0)); //yellow
	glGenBuffers(1, &sphere_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, sphere_buffer);	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3)*(sphere_point.size()+sphere_color.size()+ sphere_normal.size()), NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(point3)*sphere_point.size(), &sphere_point[0]);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(point3)*sphere_point.size(),
			sizeof(color3)*sphere_color.size(), &sphere_color[0]);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(vec3)*(sphere_point.size() + sphere_color.size()),
			sizeof(vec3)*sphere_normal.size(), &sphere_normal[0]);


		//Floor buffer
		readVertices("Floor2.txt", floor_point, floor_normal, floor_textCoord);
		floor_color.assign(floor_point.size(), color3(0.0, 1.0, 0.0)); //yellow
		

		//Sending data to floor Buffer
		glGenBuffers(1, &floor_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, floor_buffer);
		glBufferData(GL_ARRAY_BUFFER, 
				sizeof(vec3)*(floor_point.size() +
								floor_color.size()+
								floor_textCoord.size())+
					sizeof(vec2)*floor_textCoord.size(),
						NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(point3)*floor_point.size(), &floor_point[0]);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(point3)*floor_point.size(),
			sizeof(color3)*floor_color.size(), &floor_color[0]);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(vec3)*(floor_point.size()+floor_color.size()),
			sizeof(vec3)*floor_normal.size(), &floor_normal[0]);
		glBufferSubData(GL_ARRAY_BUFFER, 
						sizeof(vec3)*(floor_point.size() +
									  floor_color.size() +
									  floor_textCoord.size()),
						sizeof(vec2)*floor_textCoord.size(),
						&floor_textCoord[0]
						);
		//Axis buffer
		glGenBuffers(1, &axis_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, axis_buffer);
		glBufferData(GL_ARRAY_BUFFER,
			(sizeof(point3) + sizeof(color3))*axis_NumVertices,
			NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0,
			sizeof(point3) * 6, axis_points);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(point3) * axis_NumVertices,
			sizeof(color3) * 6, axis_color);

		//fireworks
		for (int i = 0; i < 300; i++)
		{
			/*Setting each particle velocity*/
			velocitys[i].x = 2.0*((rand() % 256) / 256.0 - 0.5);
			velocitys[i].z = 2.0*((rand() % 256) / 256.0 - 0.5);
			velocitys[i].y = 1.2*2.0*((rand() % 256) / 256.0);
			/*Setting each particle color*/
			colors[i].x = (rand() % 256) / 256.0;
			colors[i].y = (rand() % 256) / 256.0;
			colors[i].z = (rand() % 256) / 256.0;

			//std::cout << velocitys[i].x << " " << velocitys[i].y << " " << velocitys[i].z << std::endl;
		}
		/*建立VBO*/
		glGenBuffers(1, &fireworks_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, fireworks_buffer);


		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3)*(colors.size() + velocitys.size()), NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3)*velocitys.size(), &velocitys[0]);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(vec3)*velocitys.size(), sizeof(vec3)*colors.size(), &colors[0]);
}

void SphereVAO()
{
	glBindBuffer(GL_ARRAY_BUFFER, sphere_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
		BUFFER_OFFSET(sizeof(point3)*sphere_point.size()));
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0,
		BUFFER_OFFSET(sizeof(vec3)*(sphere_point.size() + sphere_color.size())));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
}

void ShadowVAO()
{
	glBindBuffer(GL_ARRAY_BUFFER, sphere_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(0);
}

void FloorVAO()
{
	glBindBuffer(GL_ARRAY_BUFFER, floor_buffer);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
		BUFFER_OFFSET(sizeof(point3)*floor_point.size()));
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0,
		BUFFER_OFFSET(sizeof(vec3)*(floor_point.size() + floor_color.size())));
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0,
		BUFFER_OFFSET(sizeof(vec3)*(floor_point.size() + floor_color.size() + floor_normal.size()) )
		);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);
}

void AxisVAO()
{
	glBindBuffer(GL_ARRAY_BUFFER, axis_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
		BUFFER_OFFSET(sizeof(point3) * axis_NumVertices));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
}

////------------NOT USED IN MAC------------------///
//void cinfigureVAO()
//{
//	//Configure Sphere Attribute
//	glGenVertexArrays(1, &sphere_VAO);
//	glBindVertexArray(sphere_VAO);
//	glBindBuffer(GL_ARRAY_BUFFER, sphere_buffer);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
//		BUFFER_OFFSET(sizeof(point3)*sphere_point.size()));
//	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0,
//		BUFFER_OFFSET(sizeof(vec3)*(sphere_point.size() + sphere_color.size())));
//	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);
//	glEnableVertexAttribArray(2);
//	glBindVertexArray(0);
//
//	//Shadow 
//	glGenVertexArrays(1, &shadow_VAO);
//	glBindVertexArray(shadow_VAO);
//	glBindBuffer(GL_ARRAY_BUFFER, sphere_buffer);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
//	glEnableVertexAttribArray(0);
//	glBindVertexArray(0);
//
//	//Configure Floor Attribute, Floor VAO
//	glGenVertexArrays(1, &floor_VAO);
//	glBindVertexArray(floor_VAO);
//	glBindBuffer(GL_ARRAY_BUFFER, floor_buffer);
//
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
//		BUFFER_OFFSET(sizeof(point3)*floor_point.size()));
//	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0,
//		BUFFER_OFFSET(sizeof(vec3)*( floor_point.size()+floor_color.size() )));
//	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);
//	glEnableVertexAttribArray(2);
//	glBindVertexArray(0);
//
//	//configure the axis
//	glGenVertexArrays(1, &axis_VAO);
//	glBindVertexArray(axis_VAO);
//		glBindBuffer(GL_ARRAY_BUFFER, axis_buffer);
//		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
//		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,
//			BUFFER_OFFSET(sizeof(point3) * axis_NumVertices));
//		glEnableVertexAttribArray(0);
//		glEnableVertexAttribArray(1);
//	glBindVertexArray(0);
//	
//}



// OpenGL initialization
void init()
{

	//init the animation 
	startPostion = posA;
	currentTarget = posB;
	rollingDir = normalize(posB - posA);
	rollingDis = length(posA - posB);
	accumRota = identity();
	rollingAxis = cross(rollingDir, vec4(0, -1, 0, 0));
	angle = 0;	

	sendingData();
	////------------NOT USED IN MAC------------------///
	//cinfigureVAO();
	
	image_set_up();

	glGenTextures(1, &texture_2D);
	glBindTexture(GL_TEXTURE_2D, texture_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ImageWidth, ImageHeight,
		0, GL_RGBA, GL_UNSIGNED_BYTE, &Image);

	glGenTextures(1, &texture_1D);
	glBindTexture(GL_TEXTURE_1D,texture_1D);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, stripeImageWidth, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, &stripeImage);
 // Load shaders and create a shader program (to be used in display())
	program = InitShader("vshader42.glsl", "fshader42.glsl");
	noLightProgram = InitShader("vAxisShading.glsl", "fAxisShading.glsl");
	fireWorksProgram = InitShader("vfireworks.glsl", "ffireworks.glsl");

	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0f);
	glLineWidth(2.0);
 
//
	ADD(MENU_SELECT, TEXT_WORLD);
	
	//Firework clock initialize
	clock.loop_num = 0;
	clock.animate_length = 100000;
	clock.clock_start_point = 0;
}

/*Processing fireworks effects*/
	/*建立单个粒子*/

	/*Caculate velocity and color*/


void openFireWorks( mat4 lookat, mat4 projection)
{

	glUseProgram(fireWorksProgram);
	setUniform(fireWorksProgram, "lookat", lookat);
	setUniform(fireWorksProgram, "projection", projection);

	glBindBuffer(GL_ARRAY_BUFFER, fireworks_buffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(sizeof(vec3)*velocitys.size()));
	glEnableVertexAttribArray(1);
	glPointSize(3.0);
	glDrawArrays(GL_POINTS, 0, velocitys.size());

}

//the sphere and shadow have the same drawing mode
GLenum SPHERE_MODE;
//----------------------------------------------------------------------------

vec4 background = vec4(0.529f, 0.807f, 0.92f, 1.0f);//ambient color
void display( void )
{
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glEnable(GL_FRAMEBUFFER);


	//glClearColor(0.529f, 0.807f, 0.92f, 1.0f);	
	//TODO if all the particle discard, set IS_INIT = 1
	//set the velocity time by velocity time

	glClearColor(background.x, background.y, background.z, background.w);
	vec4    at(0.0, 0.0, 0.0, 1.0);
    vec4    up(0.0, 1.0, 0.0, 0.0);
	mat4  mv = LookAt(eye, at, up);
	mat4 lookat = mv;
	mat4  p = Perspective(fovy, aspect, zNear, zFar);
	




	if (IS_CHOOSE(MENU_SELECT, LIGHT)){
		setUniform(program, "is_light", 1);
		setUniform(program, "lookat", lookat);


		dir.sendData(program,"dirLight");

		//null_point.sendData(program, "pointLights[0]");
		//null_spot.sendData(program, "spotLights[0]");

		if (IS_CHOOSE(MENU_SELECT, POINT_SOURCE))
		{
			////turn on the point light
			setUniform(program, "POINT_LIGHT_NUM", 1);

			//spot_light.sendData(program, "spotLights[0]");
		    point_light.sendData(program, "pointLights[0]");
			////turn off the spot light
			//null_spot.sendData(program, "spotLights[0]");
		}
		
		if (IS_CHOOSE(MENU_SELECT, SPOT_SOURCE))
		{
			setUniform(program, "SPOT_LIGHT_NUM", 1);
			////turn off the point light
			spot_light.sendData(program, "spotLights[0]");
			////turn on the spot light
			//null_point.sendData(program, "pointLights[0]");
		}

		if (IS_CHOOSE(MENU_SELECT, MULTI_LIGHTS)) 
		{
			DirectionLight dir2 = dir;
			 null_dir.sendData(program, "dirLight");

			////-------Put 2 Point Sources --------/////
			setUniform(program, "POINT_LIGHT_NUM", 2);
			PointLight point2 = point_light;	
				point2.position = vec4(4.0, 5.0, 4.0, 0.0);
				point2.ambient = vec4(0.02, 0.0, 0.02, 0.0);
				point2.specular = vec4(1.0, 1.0, 1.0, 1.0);
				point2.diffuse = vec4(1.0, 0.0, 1.0, 1.0);		
			point2.sendData(program, "pointLights[0]");
			//null_point.sendData(program, "pointLights[0]");

			PointLight point3 = point_light;
				point3.position = vec4(4.0, 5.0, -4.0, 0.0);
				point3.ambient = vec4(0.02, 0.02, 0.0, 0.0);
				point3.specular = vec4(1.0, 1.0, 1.0, 1.0);
				point3.diffuse = vec4(1.0, 1.0, 0.0, 1.0);	
			point3.sendData(program, "pointLights[1]");

			////-------Put 2 Spot Sources --------/////
			setUniform(program, "SPOT_LIGHT_NUM", 2);
			SpotLight point4 = spot_light;
			vec4 p4_dir = normalize( point4.position - point4.direction );
			GLfloat a = 2* M_PI*t/360.0;
			point4.position =  Rotate(a, p4_dir)*Translate(1.0,0.0,0.0)*vec4(-4.0, 5.0, -4.0, 0.0);
				point4.direction = vec4(0.0, 0.0, 0.0, 0.0);		
				point4.ambient = vec4(0.05, 0.05, 0.05, 1.0);
				point4.specular = vec4(1.0, 1.0, 1.0, 1.0);
				point4.diffuse = vec4(sin(a), 0.0, 0.0, 1.0);
			point4.sendData(program, "spotLights[0]");

			SpotLight point5 = spot_light;
			point5.position = Rotate(a, p4_dir)*Translate(-1.0, 0.0, 0.0)*vec4(-4.0, 5.0, -4.0, 0.0);
				point5.direction = vec4(0.0, 0.0, 0.0, 0.0);
				point5.ambient = vec4(0.05, 0.05, 0.05, 1.0);
				point5.specular = vec4(0.0, 0.0, 0.0, 1.0);
				point5.diffuse = vec4(0, 0, 0.0, 1.0);
			point5.sendData(program, "spotLights[1]");

			background = vec4(0.0f, 0.278f, 0.391f, 1.0f)+dir2.ambient+point2.ambient + point3.ambient + point4.ambient + point5.ambient;
			
		}
		
		else
		{
			background = vec4(0.529f, 0.807f, 0.92f, 1.0f);
		}

		if (IS_CHOOSE(MENU_SELECT, FLATSHADING)) {
			//std::cout << IS_CHOOSE(MENU_SELECT, FLATSHADING) << std::endl;
			setUniform(program, "SHADING_MODE", 0);
		}
		else if(IS_CHOOSE(MENU_SELECT, SMOOTHSHADING)) {
			setUniform(program, "SHADING_MODE", 1);
			//std::cout << IS_CHOOSE(MENU_SELECT, FLATSHADING) << std::endl;
		}
	}
	else{
		setUniform(program, "is_light", 0);
		background = vec4(0.529f, 0.807f, 0.92f, 1.0f);
	}

	//Both shaders use the same projection matrix p
	setUniform(program, "projection", p);
	setUniform(program, "lookat", lookat);

	/*---------------Process the fog entries--------------------*/
	if (IS_CHOOSE(MENU_SELECT, LINEAR))
		setUniform(program, "FOG_MODE", LINEAR);
	else if(IS_CHOOSE(MENU_SELECT, EXPONENTIAL))
		setUniform(program, "FOG_MODE", EXPONENTIAL);
	else if(IS_CHOOSE(MENU_SELECT, EXPONENTIA_SQUARE))
		setUniform(program, "FOG_MODE", EXPONENTIA_SQUARE);
	else
		setUniform(program, "FOG_MODE", NO_FOG);



	/*----------------------Draw the objects-------------------------*/
	//draw floor
	//disallow writting into the z-buffer
	glUseProgram(program);
	setUniform(program, "material_ambient", floor_ambient);
	setUniform(program, "material_diffuse", floor_diffuse);
	setUniform(program, "material_specular", floor_specular);
	setUniform(program, "shininess", floor_shininess);

	setUniform(program, "OBJECT_TYPE", 0);
	/*--------Setting Floor Texture -----------*/
	if (IS_CHOOSE(MENU_SELECT, Texture_Mapped_Ground)) 
			setUniform(program, "TEXTURE_MODE", 2);
	else
		setUniform(program, "TEXTURE_MODE", 0);

		glClear(GL_DEPTH_BUFFER_BIT);
		glDepthMask(GL_FALSE);
		mv = LookAt(eye, at, up);
		setUniform(program, "model_view", mv);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		FloorVAO();
		glDrawArrays(GL_TRIANGLES, 0, floor_point.size());
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

	////using shadow_program to draw
	//when the camera is above the floor, draw the shadow
	////if need to draw shadow
		if(eye.y>=0)
		{
			glUseProgram(program);

			mv = LookAt(eye, at, up)*shadow_matrix*
				Translate(startPostion + currentDis*rollingDir)*Rotate(angle, rollingAxis)*accumRota;
			setUniform(program, "model_view", mv);

			glPolygonMode(GL_FRONT_AND_BACK, SPHERE_MODE);
			ShadowVAO();
				if (IS_CHOOSE(MENU_SELECT, BLEND_SHADOW)){
					setUniform(program, "OBJECT_TYPE", 2);
					glEnable(GL_BLEND);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					glDrawArrays(GL_TRIANGLES, 0, sphere_point.size());
					glDisable(GL_BLEND);
				}
				else if(IS_CHOOSE(MENU_SELECT, SHADOW)) {
					setUniform(program, "OBJECT_TYPE", 2);
					glDrawArrays(GL_TRIANGLES, 0, sphere_point.size());
				}
		}
		glDepthMask(GL_TRUE);

	////using program to draw
	//writting floor depth data into the z-buffer, disallow the color
	glUseProgram(program);
	setUniform(program, "OBJECT_TYPE", 0);
	/*--------Setting Floor Texture -----------*/
	if (IS_CHOOSE(MENU_SELECT, Texture_Mapped_Ground))
		setUniform(program, "TEXTURE_MODE", 2);
	else
		setUniform(program, "TEXTURE_MODE", 0);
	///////////Redraw z to z-buffer/////////////////
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	mv = LookAt(eye, at, up);
	setUniform(program, "model_view", mv);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	FloorVAO();
	glDrawArrays(GL_TRIANGLES, 0, floor_point.size());

	////////////open the color mask//////////////////
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);	

	//draw the Sphere
	//check what kind of sphere
	if(IS_CHOOSE(MENU_SELECT, WIRE_FRAME))
		SPHERE_MODE = GL_LINE;
	else
		SPHERE_MODE = GL_FILL;

	////////Settiing Sphere Properties/////////
	setUniform(program, "material_ambient", sphere_ambient);
	setUniform(program, "material_diffuse", sphere_diffuse);
	setUniform(program, "material_specular", sphere_specular);
	setUniform(program, "shininess", sphere_shininess);
	
	setUniform(program, "OBJECT_TYPE", 1);
	if (IS_CHOOSE(MENU_SELECT, CHECKER_BOARD_SPHERE))
		setUniform(program, "TEXTURE_MODE", 2);
	else if (IS_CHOOSE(MENU_SELECT, COUNTER_LINE_SPHERE))
		setUniform(program, "TEXTURE_MODE", 1);
	else
		setUniform(program, "TEXTURE_MODE", 0);

		mv = LookAt(eye, at, up)*
			Translate(startPostion + currentDis*rollingDir)*Rotate(angle, rollingAxis)*accumRota;
		setUniform(program, "model_view", mv);
		glPolygonMode(GL_FRONT_AND_BACK,SPHERE_MODE);
		SphereVAO();
		glDrawArrays(GL_TRIANGLES, 0, sphere_point.size());

	//Draw the axis
	mv = LookAt(eye, at, up);
	setUniform(noLightProgram, "model_view", mv);
	setUniform(noLightProgram, "projection", p);
	glUseProgram(noLightProgram);
	AxisVAO();
	for (int i = 0; i <= 4; i += 2)
	{
		glDrawArrays(GL_LINES, i, 2);
	}

	//----Process Fireworks Event-----//
	if (IS_CHOOSE(MENU_SELECT, OPEN_FIREWORKS))
	{
		clock.run_time_elapse = glutGet(GLUT_ELAPSED_TIME);
		clock.clock = clock.run_time_elapse - clock.clock_start_point;
		clock.fireworks_time = clock.clock - clock.loop_num*clock.animate_length;
		if (clock.fireworks_time >= clock.animate_length)
		{
			clock.fireworks_time = 0;
			clock.loop_num++;
		}

		openFireWorks(lookat, p);
		setUniform(fireWorksProgram, "t", clock.fireworks_time);
	}
	else
	{
		clock.clock_start_point = glutGet(GLUT_ELAPSED_TIME);
		clock.loop_num = 0;
		clock.fireworks_time = 0;
	}
	
	glutSwapBuffers();
}
int commandCount = 0;
int i = 0;
float tr = 0;
int readInterval = 8;
//---------------------------------------------------------------------------
void idle(void)
{
	//eye = startPostion + currentDis*rollingDir;
	t += 0.1;
	tr += 0.001;
	if (t >= 180)
		t = 0;
	//std::cout << tr << std::endl;
	if (tr >= readInterval&&commandCount<commandList.size())
	{
		static int pre_command;
		static int pre_speed;
		int command = commandList[commandCount][0];
		speed = commandList[commandCount][1];
		commandCount++;
		if (command)
			ADD(MENU_SELECT, command);
		else
		{
			DELETE(MENU_SELECT, pre_command);
			pre_command = command;
			pre_speed = speed;
		}
		tr = 0;
		//speed = 1;
		////readCommand("Commands.txt", speed);	
		//if (inFile.good())
		//{
		//	std::cout << "Read::";
		//	static int pre_command;
		//    int command;
		//	static int pre_speed;
		//	char blank;
		//	inFile >> command;
		//	//speed = 0.4f;
		//	std::cout << command << std::endl;
		//	//std::cout << command << std::endl;
		//	//unsigned int COMMAND = commands[command];
		//	if (command)
		//		ADD(MENU_SELECT, command);
		//	else
		//	{
		//		DELETE(MENU_SELECT, pre_command);
		//		pre_command = command;
		//		pre_speed = speed;
		//	}
		//	tr = 0;
		//}
	}

	if (currentDis + d >= rollingDis)
	{
		accumRota = Rotate(angle, rollingAxis)*accumRota;
		angle = 0;
		currentDis = 0;
		if (i == 0) {
			i = 1;
			startPostion = posB;
			currentTarget = posC;		
		}
		else if (i == 1) {
			i = 2;
			startPostion = posC;
			currentTarget = posA;
		}
		else if(i==2){
			i = 0;
			startPostion = posA;
			currentTarget = posB;
		}
		setRollingStatus(startPostion, currentTarget, rollingDir, rollingDis, rollingAxis);
	}
	angle += 0.05*speed;
	currentDis = 2 * M_PI*angle / 360.0f;
    glutPostRedisplay();
}
//----------------------------------------------------------------------------
int flag = 0;
void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
	case 033: // Escape Key
	case 'q': case 'Q':
	    exit( EXIT_SUCCESS );
	    break;

        case 'X': eye[0] += 1.0; break;
	case 'x': eye[0] -= 1.0; break;
        case 'Y': eye[1] += 1.0; break;
	case 'y': eye[1] -= 1.0; break;
        case 'Z': eye[2] += 1.0; break;
	case 'z': eye[2] -= 1.0; break;

	case 'o':case'O':
		DELETE(MENU_SELECT, TEXT_EYE);
		ADD(MENU_SELECT, TEXT_WORLD);
		setUniform(program, "TEXTURE_FRAME", 0);
		break;
	case 'e':case'E':
		DELETE(MENU_SELECT, TEXT_WORLD);
		ADD(MENU_SELECT, TEXT_EYE);
		setUniform(program, "TEXTURE_FRAME", 1);
		break;
	
	case'v':case'V':
			setUniform(program, "MAPPING_MODE", 0);
		break;

	case 's':case 'S':
			setUniform(program, "MAPPING_MODE", 1);
		break;
		
	case 'u':case'U':
		if (IS_CHOOSE(MENU_SELECT, TEXT_WORLD))
		{
			setUniform(program, "LATTICE_MODE", 2);
			setUniform(program, "MAPPING_MODE", 0);
			flag = 1;
		}
		break;
	
	case 't':case'T':
		if (IS_CHOOSE(MENU_SELECT, TEXT_WORLD))
		{
			setUniform(program, "LATTICE_MODE", 1);
			setUniform(program, "MAPPING_MODE", 1); 
			flag = 1;
		}
		break;
	
	case 'l':case'L':
		if (flag == 0) flag = 1;
		else flag = 0;
		setUniform(program, "LATTICE_MODE", flag);
		break;
	
	
	case 'b': case'B': // Toggle between animation and non-animation
		   animationFlag = 1;
		   if (beginRollingFlag == 1)
			   glutIdleFunc(idle);
            break;
    }
    glutPostRedisplay();
}
//----------------------------------------------------------------------------
void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		if (animationFlag == 1) {
			beginRollingFlag = 1 - beginRollingFlag;
			if (beginRollingFlag == 1) glutIdleFunc(idle);
			else                    glutIdleFunc(NULL);
		}

	}
}
//----------------------------------------------------------------------------
void reshape(int width, int height)
{
    glViewport(0, 0, width, height);
    aspect = (GLfloat) width  / (GLfloat) height;
    glutPostRedisplay();
}
//----------------------------------------------------------------------------
void processMenuEvent(int option)
{
	switch (option)
	{
	case DEFAULT_VIEW_POINT:
		eye = init_eye;
		beginRollingFlag = 1;
		if (animationFlag == 1)
			glutIdleFunc(idle);
		break;
	case QUIT:
		exit(EXIT_SUCCESS);
		break;
	}
}

void createGLUTMenus() {
	
	int wire_submenu = createSubMenu_YN(processWireFrameMenuEvent);
	int light_submenu = createSubMenu_YN(processLightMenuEvent);
	int shading_submenu = createSubMenu_YN(processShadingMenuEvent,"Flat Shading", "Smooth Shading");
	int light_source = createSubMenu_YN(processLightSourceMenuEvent, "Point Light", "Spot Light");
	int multi_lights = createSubMenu_YN(processMultiLightsMenuEvent);
	int tex_ground = createSubMenu_YN(processGroundTextrueEvent);
	int firework = createSubMenu_YN(processFireWorkEvent);

	/*---------------------the shadow menu-----------------*/
	int shadow_submenu = glutCreateMenu(processShadowMenuEvent);
	glutAddMenuEntry("No shadow", NO);
	glutAddMenuEntry("Shadow", SHADOW);
	glutAddMenuEntry("Blend Shadow", BLEND_SHADOW);
	glutAttachMenu(GLUT_LEFT_BUTTON);

	/*------------------the fog menu-----------------------*/
	int fog_menu = glutCreateMenu(processFogMenuEvent);
	glutAddMenuEntry("no fog", NO_FOG);
	glutAddMenuEntry("linear", LINEAR);
	glutAddMenuEntry("exponential", EXPONENTIAL);
	glutAddMenuEntry("exponential square", EXPONENTIA_SQUARE);
	glutAttachMenu(GLUT_LEFT_BUTTON);

	/*----------------the sphere texture menu--------------------*/
	int sphere_texture_menu = glutCreateMenu(processSphereTextureEvent);
	glutAddMenuEntry("no texture", NO_Mapped_SPHERE);
	glutAddMenuEntry("Counter Line sphere", COUNTER_LINE_SPHERE);
	glutAddMenuEntry("Checker board sphere", CHECKER_BOARD_SPHERE);
	glutAttachMenu(GLUT_LEFT_BUTTON);

	int menu = glutCreateMenu(processMenuEvent);
	glutAddMenuEntry("Default View Point", DEFAULT_VIEW_POINT);
	glutAddSubMenu("Shadow", shadow_submenu);
	glutAddSubMenu("Wire Frame", wire_submenu);
	glutAddSubMenu("Light", light_submenu);
	glutAddSubMenu("Shading", shading_submenu);
	glutAddSubMenu("Light Source", light_source);
	glutAddSubMenu("Multi Lights", multi_lights);
	glutAddSubMenu("Fog", fog_menu);
	glutAddSubMenu("Texture Mapped Ground", tex_ground);
	glutAddSubMenu("Texture Mapped Sphere", sphere_texture_menu);
	glutAddSubMenu("Fireworks", firework);
	glutAddMenuEntry("Quit", QUIT);

	glutAttachMenu(GLUT_LEFT_BUTTON);
}


//----------------------------------------------------------------------------
int main(int argc, char **argv)
{ int err;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(512, 512);

     //glutInitContextVersion(3, 2);
     //glutInitContextProfile(GLUT_CORE_PROFILE);
    glutCreateWindow("ver90dg");  

  /* Call glewInit() and error checking */
	err = glewInit();
    commandList = readCommand("Commands/line0dg/control_flow.txt");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
	glutIdleFunc(idle);
	ADD(MENU_SELECT, LIGHT);

    init();
    glutMainLoop();
	//inFile.close();
    return 0;
}
