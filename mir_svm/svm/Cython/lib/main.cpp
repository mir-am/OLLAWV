#include <iostream>
#include "FileReader.h"
<<<<<<< HEAD
=======
#include "svm.h"
>>>>>>> Converting CSV files to dense LIBSVM format.

using namespace std;

int main()
{

<<<<<<< HEAD
    FileReader data("../../../dataset/pima-indian.csv");
    data.readDataFile(true);
=======
    SVMProblem prob;

    FileReader data("../../../dataset/pima-indian.csv");
    data.readDataFile(true);
    data.toLIBSVM(prob);
    data.printData(prob);
>>>>>>> Converting CSV files to dense LIBSVM format.

    cout << "Hello world!" << endl;
    return 0;
}
