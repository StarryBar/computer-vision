#pragma once
#include "Angel-yjc.h"
#include"LoadData.h"
#include "CommandCode.h"
#include "rotate-cube-new.h"


extern GLuint MENU_SELECT;

int createSubMenu_YN(void(*func)(int value), const char* yes = "Yes", const char* no = "No");


void processShadowMenuEvent(int option);
void processWireFrameMenuEvent(int option);
void processLightMenuEvent(int option);
void processLightSourceMenuEvent(int option);
void processMultiLightsMenuEvent(int option);

void processShadingMenuEvent(int option);

void processFogMenuEvent(int option);
void processGroundTextrueEvent(int option);
void processSphereTextureEvent(int option);
void processFireWorkEvent(int option);