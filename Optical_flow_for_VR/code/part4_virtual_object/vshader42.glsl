layout(location = 0) in  vec3 vPosition;
layout(location = 1) in  vec3 vcolor;
layout(location = 2) in  vec3 vnormal;
layout(location = 3) in  vec2 vtextCoord;
out vec4 color;
out vec4 world_frame_position;
out vec4 eye_frame_position;
out vec2 textCoord;

uniform mat4 lookat;
uniform mat4 model_view;
uniform mat4 projection;

uniform int is_light;

/*Shading Modes*/
#define FLAT 0
#define SMOOTH 1
uniform int SHADING_MODE;

/*Object's type*/
#define SHADOW 2
#define SPHERE 1
#define FLOOR 0
uniform int OBJECT_TYPE;

struct DirLight {
    vec4 direction;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

struct PointLight{
    vec4 position;
    vec3 attenuation;
   
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
}; 

struct SpotLight{
    vec4 direction;
    vec4 position;
    vec3 attenuation;
    
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;

    float max_angle_half;
    float e;
};

struct AmbientLight{
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

uniform vec4 material_ambient;
uniform vec4 material_diffuse;
uniform vec4 material_specular;

uniform DirLight dirLight;
uniform float shininess;

vec4 caculateDirLight(DirLight dirLight, vec4 N, vec4 E)
{
    vec4 L = normalize(-dirLight.direction);
    vec4 H = normalize(L+E);
    
    vec4 Ra = material_ambient;
    vec4 Rd = max(0.0, dot(L,N))*material_diffuse;
    float spec =  pow(max(0.0, dot(N,H)), shininess);
    vec4 Rs = spec*material_specular;

    vec4 ambient = Ra*dirLight.ambient;
    vec4 diffuse = Rd*dirLight.diffuse;
    vec4 specular = Rs*dirLight.specular;

    return (ambient + diffuse + specular);
}

//all the point light sources
#define MAX_POINT_LIGHT_NUM  10
uniform int POINT_LIGHT_NUM;
uniform PointLight pointLights[MAX_POINT_LIGHT_NUM];
vec4 caculatePointLight(PointLight pointLight,vec4 ePos ,vec4 N, vec4 E)
{
    pointLight.position = lookat*pointLight.position;
    vec4 L = normalize(pointLight.position-ePos);
    vec4 H = normalize(L+E);
    
    float d = length(pointLight.position-ePos);
    vec3 vd = vec3(1,d,pow(d,2));
    float att = 1.0/dot(pointLight.attenuation,vd);

    vec4 Ra = material_ambient;
    vec4 Rd = max(0.0, dot(L,N))*material_diffuse;
    float spec =  pow(max(0.0, dot(N,H)), shininess);
    vec4 Rs = spec*material_specular;

    vec4 ambient = Ra*pointLight.ambient;
    vec4 diffuse = Rd*pointLight.diffuse;
    vec4 specular = Rs*pointLight.specular;

    return att*(ambient + diffuse + specular);
}
//all the spot light sources
#define MAX_SPOT_LIGHT_NUM   10
uniform int SPOT_LIGHT_NUM;
uniform SpotLight spotLights[MAX_SPOT_LIGHT_NUM];
vec4 caculateSpotLight(SpotLight spotLight, vec4 ePos ,vec4 N, vec4 E)
{
    spotLight.position = lookat*spotLight.position;
    spotLight.direction =lookat*spotLight.direction;

    vec4 L = normalize(spotLight.position-ePos);
    vec4 H = normalize(L+E);
    
    
    vec4 pv = normalize(ePos-spotLight.position);
    vec4 ps = normalize(spotLight.direction-spotLight.position);
    float cos_max_angle = cos(radians(spotLight.max_angle_half));
    float cos_angle = dot(pv,ps);
    float k;
    if(cos_angle>=cos_max_angle)  
        k=pow(cos_angle,spotLight.e);
    else
        k=0.0;

    float d = length(spotLight.position-ePos);
    vec3 vd = vec3(1,d,pow(d,2));
    float att = k/dot(spotLight.attenuation,vd);
        
    vec4 Ra = material_ambient;
    vec4 Rd = max(0.0, dot(L,N))*material_diffuse;
    float spec =  pow(max(0.0, dot(N,H)), shininess);
    vec4 Rs = spec*material_specular;

    vec4 ambient = Ra*spotLight.ambient;
    vec4 diffuse = Rd*spotLight.diffuse;
    vec4 specular = Rs*spotLight.specular;

    return att*(ambient + diffuse + specular);
}

/*Here should provide 4 kind of light: ambient, direction, point and spot*/
void main() 
{
    vec4 ePos = model_view*vec4(vPosition,1.0);
    vec4 E = normalize(-ePos);
    vec4 N;
    if(SHADING_MODE == SMOOTH && OBJECT_TYPE == SPHERE)
        N = normalize( model_view*vec4(vPosition,0.0)); 
    else
        N = normalize( model_view*vec4(vnormal,0.0));

    vec4 dir_color = caculateDirLight(dirLight, N, E);
    vec4 spot_color=vec4(0.0,0.0,0.0,0.0);
    vec4 point_color=vec4(0.0,0.0,0.0,0.0);
    for(int i=0; i<POINT_LIGHT_NUM;i++)
    {
        point_color =point_color+ caculatePointLight(pointLights[i],ePos,N,E);
    }
   
   for(int i=0; i<SPOT_LIGHT_NUM;i++)
    {
        spot_color = spot_color+ caculateSpotLight(spotLights[i], ePos, N, E);
    }
    
    vec4 global_color = vec4(1.0,1.0, 1.0, 1.0)*vec4(0.2, 0.2, 0.2, 1.0); 

    if(is_light==1)
        color =global_color+ dir_color+spot_color+point_color;
    else 
        color = vec4(vcolor,1.0);

    eye_frame_position = model_view* vec4(vPosition,1.0);
    world_frame_position = vec4(vPosition,1.0);
    textCoord = vtextCoord;

    gl_Position = projection * model_view * vec4(vPosition,1.0);
} 
