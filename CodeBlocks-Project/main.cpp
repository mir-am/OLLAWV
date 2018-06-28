#include <iostream>
#include <vector>
#include <algorithm>
#include "timer.h"

using namespace std;

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
    timeElasped.start();

    cout << "Number of arguments: "  << argc << endl;
    cout << "Program name:" << argv[0] << endl;



    timeElasped.stop();

    cout << "Elapsed: " << timeElasped.getTimeElapsed() * 1000 << "ms" << endl;


    return 0;
}
