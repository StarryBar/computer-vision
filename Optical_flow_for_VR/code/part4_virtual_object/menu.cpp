#include"menu.h"

GLuint MENU_SELECT  =POINT_SOURCE;

int createSubMenu_YN(void(*func)(int value), const char* yes, const char* no)
{
	int submenu = glutCreateMenu(func);
	glutAddMenuEntry(yes, YES);
	glutAddMenuEntry(no, NO);
	glutAttachMenu(GLUT_LEFT_BUTTON);
	return submenu;
}


void processShadowMenuEvent(int option)
{
	switch (option)
	{
		case SHADOW:
			ADD(MENU_SELECT,SHADOW);
			DELETE(MENU_SELECT, BLEND_SHADOW);
			break;
		case BLEND_SHADOW:
			ADD(MENU_SELECT, BLEND_SHADOW);
			DELETE(MENU_SELECT, SHADOW);
			break;
		case NO:
			DELETE(MENU_SELECT, SHADOW);
			DELETE(MENU_SELECT, BLEND_SHADOW);
			break;
		default:
			break;
	}
}
void processWireFrameMenuEvent(int option)
{
	switch (option)
	{
	case YES:
		MENU_SELECT |= WIRE_FRAME;
		break;
	case NO:
		MENU_SELECT &= (~WIRE_FRAME);
		break;
	default:
		break;
	}
}

void processLightMenuEvent(int option)
{
	MENU_SELECT &= (~POINT_SOURCE);
	MENU_SELECT &= (~SPOT_SOURCE);
	MENU_SELECT &= (~MULTI_LIGHTS);
	switch (option)
	{
	case YES:
		MENU_SELECT |= LIGHT;
		break;
	case NO:
		MENU_SELECT &= (~LIGHT);
		break;
	default:
		break;
	}
}

void processLightSourceMenuEvent(int option)
{
	MENU_SELECT &= (~MULTI_LIGHTS);
	if (IS_CHOOSE(MENU_SELECT, LIGHT))
	{
		switch (option)
		{
		case YES:
			MENU_SELECT |= POINT_SOURCE;
			MENU_SELECT &= (~SPOT_SOURCE);
			std::cout <<"Point: "<< MENU_SELECT << std::endl;
			break;
		case NO:
			MENU_SELECT |= SPOT_SOURCE;
			MENU_SELECT &= (~POINT_SOURCE);
			std::cout << "Spot: " << MENU_SELECT << std::endl;
			break;
		default:
			break;
		}
	}
}

void processMultiLightsMenuEvent(int option)
{
	if (IS_CHOOSE(MENU_SELECT, LIGHT))
	{
		switch (option)
		{
		case YES:
			ADD(MENU_SELECT, MULTI_LIGHTS);
			DELETE(MENU_SELECT, POINT_SOURCE);
			DELETE(MENU_SELECT, SPOT_SOURCE);
			break;
		case NO:
			ADD(MENU_SELECT, LIGHT);
			DELETE(MENU_SELECT, MULTI_LIGHTS);
			break;
		default:
			break;
		}
	}
}

void processShadingMenuEvent(int option)
{
	DELETE(MENU_SELECT, SMOOTHSHADING);
	DELETE(MENU_SELECT, FLATSHADING);
	if (IS_CHOOSE(MENU_SELECT, LIGHT) && (!IS_CHOOSE(MENU_SELECT, WIRE_FRAME)) )
	{
		switch (option)
		{
		case YES:
			ADD(MENU_SELECT, FLATSHADING);
			break;
		case NO:
			ADD(MENU_SELECT, SMOOTHSHADING);
			break;
		default:
			break;
		}
	}
}

void processFogMenuEvent(int option)
{
		DELETE(MENU_SELECT, NO_FOG);
		DELETE(MENU_SELECT, LINEAR);
		DELETE(MENU_SELECT, EXPONENTIAL);
		DELETE(MENU_SELECT, EXPONENTIA_SQUARE);

		switch (option)
		{
			case NO_FOG:
				ADD(MENU_SELECT, NO_FOG);
				break;
			case LINEAR:
				ADD(MENU_SELECT, LINEAR);
				std::cout << "linear" << std::endl;
				break;
			case EXPONENTIAL:
				ADD(MENU_SELECT, EXPONENTIAL);
				std::cout << "exponential" << std::endl;
				break;
			case EXPONENTIA_SQUARE:
				ADD(MENU_SELECT, EXPONENTIA_SQUARE);
				std::cout << "exponential square" << std::endl;
				break;
		}
	
}

void processGroundTextrueEvent(int option)
{
	switch (option)
	{
		case YES:
			ADD(MENU_SELECT, Texture_Mapped_Ground);
			break;
		case NO:
			DELETE(MENU_SELECT, Texture_Mapped_Ground);
			break;
	}
}

void processSphereTextureEvent(int option)
{
	DELETE(MENU_SELECT, CHECKER_BOARD_SPHERE);
	DELETE(MENU_SELECT, COUNTER_LINE_SPHERE);
	switch (option)
	{
	case NO_Mapped_SPHERE:
		break;
	case COUNTER_LINE_SPHERE:
		ADD(MENU_SELECT, COUNTER_LINE_SPHERE);
		break;
	case CHECKER_BOARD_SPHERE:
		ADD(MENU_SELECT, CHECKER_BOARD_SPHERE);
		break;
	default:
		break;
	}
}

void processFireWorkEvent(int option)
{
	switch (option)
	{
	case YES:
		ADD(MENU_SELECT, OPEN_FIREWORKS);
		break;
	case NO:
		DELETE(MENU_SELECT, OPEN_FIREWORKS);
		break;
	}
}
