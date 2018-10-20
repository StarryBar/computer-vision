in vec4 color;
in vec4 position;
in vec4 world_frame_position;
in vec4 eye_frame_position;
in vec2 textCoord;
out vec4 fColor;

/*Fog modes*/
#define NO_FOG			 2048
#define EXPONENTIAL		 1024 
#define LINEAR			  512
#define EXPONENTIA_SQUARE 256
uniform int FOG_MODE;

/*Object's type*/
#define SHADOW 2
#define SPHERE 1
#define FLOOR 0
uniform int OBJECT_TYPE;

/**/
#define TEXT_WORLD 0
#define TEXT_EYE   1
uniform int TEXTURE_FRAME;

/*Shading Modes*/
#define FLAT 0
#define SMOOTH 1
uniform int SHADING_MODE;

/*Texture Modes*/
#define NO_TEXTURE 0
#define COUNTER_LINES 1
#define CHECKER_BOARD 2
uniform int TEXTURE_MODE;

/*LATTICE Modes*/
#define NO_LATTICE 0
#define TILTED     1
#define UPRIGTH    2
uniform int LATTICE_MODE;

#define VERTICAL 0
#define SLANTED  1
uniform int MAPPING_MODE;

uniform sampler2D texture_2D;
uniform sampler1D texture_1D;
uniform mat4 model_view;
uniform mat4 shadow_matrix;

vec4 fog_color = vec4(0.7, 0.7, 0.7, 0.5);
vec4 shadow_color=vec4(0.25, 0.25, 0.25, 0.65);
//vec4 shadow_color=vec4(0.25, 0.25, 0.25, 0.65);

float  FogEffect(float start, float end, vec4 pos, float density)
{
    float f;
    switch(FOG_MODE)
        {
        case LINEAR:
            f = (end - length(pos.xyz))/(end-start);
            break;
        case EXPONENTIAL:
            f = 1.0/exp(density*length(pos.xyz));
             break;
        case EXPONENTIA_SQUARE:
            f= 1.0/exp(pow(density*length(pos.xyz),2.0) );
            break;
        case NO_FOG:
            f = 1;
            break;
        }
    return f;
}

vec4 ProcessShadowColor()
{
    switch(TEXTURE_MODE)
    {
        case NO_TEXTURE:
            return  color;
        break;
        default:
            return texture(texture_2D,textCoord)*color;         
    }
}
vec4 processSphereColor(vec4 tPos)
{
    vec4 c;
    vec2 coord;
    switch(TEXTURE_MODE)
    {
        case NO_TEXTURE:
            c = color;
        break;

        case COUNTER_LINES:
            // caculate the texture coord
                if(MAPPING_MODE==VERTICAL)    //it is a vertical
                    coord.x = tPos.x*2.5;
                else                          //(it is a slanted)
                    coord.x = 1.5*(tPos.x+tPos.y+tPos.z);
            // setting color 
                c = texture(texture_1D, coord.x)*color;
                //if(c.yz == vec2(0.0,0.0))
                        //c.xyz = vec3(0.9,0.1,0.1);  
                    //c = c*color;
                break;

        case CHECKER_BOARD:
        // caculate the texture coord  
            if(MAPPING_MODE==VERTICAL){       //it is a vertical
                coord.x = 0.5*(tPos.x+1);
                coord.y = 0.5*(tPos.y+1);
            }
            else{                               //(it is a slanted)
                coord.x = 0.3*(tPos.x+tPos.y+tPos.z);
                coord.y = 0.3*(tPos.x-tPos.y-tPos.z);
            }

        // setting color 
            c = texture(texture_2D,coord);
            if(c.x==0.0)
                c.xyz = vec3(0.9,0.1,0.1);
                c *=color;  
            break;
        }

        return c;
}
vec2 ProcessLaticeTexture(vec4 position)
{
    vec2 coord;
    switch(LATTICE_MODE)
    {
        case NO_LATTICE:
            break;
        case UPRIGTH:
                coord.x = 0.5*(position.x+1);
                coord.y = 0.5*(position.y+1);
            break;

        case TILTED:
                coord.x = 0.3*(position.x+position.y+position.z);
                coord.y = 0.3*(position.x-position.y-position.z);
    break;
    }
    return coord;
}
void main() 
{ 

    float f =FogEffect(0.0, 18.0, eye_frame_position, 0.09);    
    vec2 coord;
    vec4 c;
    vec4 final_color;
    if(OBJECT_TYPE==FLOOR)
    {
        c = ProcessShadowColor();
    }
    else //the sphere and shadow has the same lattice coordinate
    {
        vec4 tPos;
        if(TEXTURE_FRAME==TEXT_EYE)    //(it is eye frame)
            tPos = eye_frame_position;
        else                           //(it is world frame)
            tPos = world_frame_position; 
        
        if(OBJECT_TYPE == SPHERE)      
         c = processSphereColor(tPos);
        else
            c=shadow_color;             

        coord = ProcessLaticeTexture(world_frame_position);       
        /*if we have lattice effect, them discard the unnecessary part*/
        if(LATTICE_MODE!=NO_LATTICE)
            if(fract(4*coord.x)<0.35 && fract(4*coord.y)<0.35)
                    discard;
    }

    //Output the color
    if(OBJECT_TYPE != SHADOW)
        fColor = mix(fog_color,c, clamp(f,0,1));
    else
    {
        if(LATTICE_MODE!=NO_LATTICE)
            if(fract(4*coord.x)<0.35 && fract(4*coord.y)<0.35)
                    discard;
        fColor =mix(fog_color, c,clamp(f,0,1));
    }

} 

