#include "ReadFromOpticalFlow.h"

using namespace std;
vector<vector<int>> readCommand(std::string fileName)
{
	int speed=1;
	vector<vector<int>> commandList;
	std::ifstream inFile(fileName, std::ios::in);
	if (!inFile)
	{
		std::cerr << "Unable to open file" << std::endl;
	}
	//delete the previous data and generate the new vertices array

	while (inFile.good())
	{
		
		int command; char blank;
		inFile >> speed >>blank>>command;
		vector<int> cs(2);
		cs[0] = pow(2,command+1);
		cs[1] = speed; 
		std::cout << speed << std::endl;
		std::cout << command << std::endl;
		commandList.push_back(cs);
	};
	inFile.close();


		return commandList;
	}
	

