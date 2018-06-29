#include "ui.h"
#include <iostream>
#include <string>
#include <stdlib.h>


void exitHelp()
{
    std::string helpMessage =
    "Usage: SVM [options] dataset file\n"
    "options:\n"
    "-c cost: set the penalty parameter of C. (default 1)\n"
    "-g gamma: the parameter of RBF kernel function. (default 1)\n"
    "-k folds: k-fold crossvalidation mode.\n"
    "-e epsilon: set tolerance of termination criterion. (default 0.1)\n"
    "-f file: name of dataset file\n";

    std::cout << helpMessage << std::endl;
    exit(1);
}

void parseCommmandLine(int& argc, char **&argv, UserInput& userIn)
{
    // Default values of SVM parameters
    userIn.parameters.C = 1;
    userIn.parameters.gamma = 1;
    userIn.parameters.e = 0.1;
    userIn.numFolds = 5;

    // Parse command line arguments [option]
    for(int i = 1; i < argc; ++i)
    {
        if((argv[i][0] != '-') || (++i >= argc))
            exitHelp();

        switch(argv[i-1][1])
        {
            case 'c':

                userIn.parameters.C = atof(argv[i]);
                break;

            case 'g':

                userIn.parameters.gamma = atof(argv[i]);
                break;

            case 'e':

                userIn.parameters.e = atof(argv[i]);
                break;

            case 'k':

                userIn.numFolds = atoi(argv[i]);

                if(userIn.numFolds < 2)
                {
                    std::cout << "k-fold cross validation: k must be >= 2" << std::endl;
                    exitHelp();
                }
                break;

            case 'f':

                userIn.dataFileName = argv[i];
                std::cout << "Filename:" << argv[i] << std::endl;
                break;

            default:

                std::cout << "Unknown option: -" << argv[i-1][1] << std::endl;
                exitHelp();
        }

    }
}
