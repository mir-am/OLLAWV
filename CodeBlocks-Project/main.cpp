#include <iostream>
#include <vector>
#include <algorithm>
#include "svm.h"
#include "ui.h"
#include "FileReader.h"
#include "timer.h"


static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t = times; t > 0; t /= 2)
	{
		if(t % 2 == 1) ret *= tmp;
		tmp = tmp * tmp;
	}

	return ret;
}

int main(int argc, char** argv)
{

    Timer timeElasped;

    UserInput userIn;
    SVMProblem userProb;
    std::string errorMsg;

    timeElasped.start();

    parseCommmandLine(argc, argv, userIn);

    // Reading LIBSVM dataset supplied by the user
    FileReader userDataset(userIn.dataFileName);
    userDataset.readDataFile(userProb);
    userDataset.readLIBSVM(userProb);

    errorMsg = checkInputParameter(userIn.parameters);

    if(errorMsg.length() != 0)
    {
        std::cout << "ERROR: " << errorMsg << std::endl;
        exit(1);
    }


    SVMModel model = trainSVM(userProb, userIn.parameters);


    timeElasped.stop();

    std::cout << "Elapsed: " << timeElasped.getTimeElapsed() * 1000 <<
    "ms" << std::endl;


    return 0;
}


