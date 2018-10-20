#pragma once

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))

#define IS_CHOOSE(MENU, ENTRY) ((MENU & ENTRY) == ENTRY)
#define ADD(MENU, ENTRY) (MENU |= ENTRY)
#define DELETE(MENU, ENTRY) (MENU &= (~ENTRY))

//menu entry
#define DEFAULT_VIEW_POINT 32
#define QUIT  64
#define YES 2
#define NO  1

/*------------Fireworks----------------*/
#define OPEN_FIREWORKS	524288
/*------------Texture Frame-----------*/
#define TEXT_EYE		262144
#define TEXT_WORLD		131072
/*-------Texture Mapped Sphere--------*/
#define NO_Mapped_SPHERE     16384
#define COUNTER_LINE_SPHERE	 32768
#define CHECKER_BOARD_SPHERE 65536

/*-------Texture Mapped Ground--------*/
#define Texture_Mapped_Ground 8192

/*-------Fog Entry-----*/
#define NO_FOG			 2048
#define EXPONENTIAL		 1024 
#define LINEAR			  512
#define EXPONENTIA_SQUARE 256

#define MULTI_LIGHTS	128  
#define SPOT_SOURCE		64	//1000000
#define POINT_SOURCE	32	//100000
#define SHADOW			16	//10000
#define BLEND_SHADOW	4096
#define LIGHT			8	//01000
#define WIRE_FRAME		4	//00100
#define FLATSHADING		2	//00010
#define SMOOTHSHADING	1	//00001
#define RETRIVE         0

/*

Event code	Function
0	Flat shading
1	Wire Frame
2	Turn Light on (Default mode)
3	Add Shadow
4	Point light on
5	Spot light on
6	Multi Light effect
7	Exponential Square Fog
8	Linear Fog
9	Exponential Fog
10	Fog off
11	Transparent Shadow

*/

