#include "misc.h"
#include <iostream>


void printData(const SVMNode *x, int numPoints)
{

// prob.x is column ordered.

    for(int i = 0; i < numPoints; ++i)
    {
        for(int j = 0; j < x[i].dim; ++j)
        {
            std::cout << x[i].values[j] << "     ";
        }

        std::cout << std::endl;

    }

}
