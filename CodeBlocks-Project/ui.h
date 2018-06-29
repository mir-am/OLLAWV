#ifndef UI_H
#define UI_H

#include "svm.h"
#include <string>

struct UserInput
{
    SVMParameter parameters;
    int numFolds;
    std::string dataFileName;

};

void exitHelp();

void parseCommmandLine(int& argc, char **&argv, UserInput& userIn);

#endif // UI_H
