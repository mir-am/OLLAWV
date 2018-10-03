#ifndef FILEREADER_H_INCLUDED
#define FILEREADER_H_INCLUDED

#include <string>
#include <vector>
#include "svm.h"

typedef std::vector<std::vector<double> > stdvecvec;
typedef std::vector<double> stdvec;

std::vector<std::string> splitString(const std::string &str, char delim=' ');

class FileReader
{
    private:

        std::string fileName;
        size_t elements; // Number of elements in dataset

        std::vector<std::vector<std::string> > trainingSet;

        //std::string delimeter;
        // A vector of vectors to store data
        //std::vector <std::vector <std::string> > dataList;

    public:

        FileReader(std::string filename):
            fileName(filename) {}

        void readDataFile(bool ignoreHeader);

        //void readLIBSVM(SVMProblem& prob);

        // Only for debugging purpose
        //void printData(SVMProblem& prob);

        // Function to fetch data from CSV file
        //void getData(bool ignoreheader=false);

        // Convert STL vector<string> to STL vector<double>
        //stdvecvec convertVecD();

        //        FileReader(std::string filename, std::string delm):
//            fileName(filename), delimeter(delm) {}


};


#endif // CSVREADER_H_INCLUDED
