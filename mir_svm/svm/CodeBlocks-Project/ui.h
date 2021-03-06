#ifndef UI_H
#define UI_H

#include "svm.h"
#include <string>

struct UserInput
{
    SVMParameter parameters;
    bool CV; // whether or not to run crossvalidation
    int numFolds;
    std::string dataFileName;
    std::string testFileName;

};

void exitHelp();

void parseCommmandLine(int& argc, char **&argv, UserInput& userIn);

std::string checkInputParameter(const SVMParameter&);

#endif // UI_H
