#include <iostream>
#include "FileReader.h"

using namespace std;

int main()
{

    FileReader data("../../../dataset/pima-indian.csv");
    data.readDataFile(true);

    cout << "Hello world!" << endl;
    return 0;
}
