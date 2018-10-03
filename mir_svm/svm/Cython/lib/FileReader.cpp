#include "FileReader.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>

// Include WIN API for error handling on Windows OS.
#ifdef _WIN32
#include <direct.h>
#endif

std::vector<std::string> splitString(const std::string &str, char delim)
{
    std::stringstream strStream(str);
    std::string item;
    std::vector<std::string> tokens;

    while(std::getline(strStream, item, delim))
    {

        // Ignore white space in lines of dataset file
        if(item.find_first_not_of(' ') != std::string::npos)
        {
            //std::cout << item << std::endl;
            tokens.push_back(item);
        }

    }

    return tokens;
}


void FileReader::readDataFile(bool ignoreHeader=false)
{
    std::ifstream file(fileName.c_str());

    if(file.is_open())
    {
        std::string line;

        // Skip first line which is header names
        if(ignoreHeader == true)
        {
            std::getline(file, line);
        }

        while(std::getline(file, line))
        {
            // Tokenize
            trainingSet.push_back(splitString(line));
        }

        std::cout << "Read File Successfully." << std::endl;


        file.close();
    }
    else
    {
        std::cout << "Failed to open dataset " << fileName << std::endl;

       exit(1);
    }


}

//void FileReader::readDataFile(SVMProblem& prob)
//{
//    elements = 0;
//    prob.l = 0;
//
//    std::ifstream file(fileName.c_str());
//
//    if(file.is_open())
//    {
//        std::cout << "Successfully read training datafile!" << std::endl;
//        std::string line;
//
//        while(std::getline(file, line))
//        {
//            // Tokenize
//            trainingSet.push_back(splitString(line));
//
//            elements += trainingSet[prob.l].size();
//            ++prob.l;
//        }
//
//        std::cout << "Samples: " << prob.l << std::endl;
//        std::cout << "Elements: " << elements << std::endl;
//
//        file.close();
//    }
//    else
//    {
//        std::cout << "Failed to open dataset " << fileName << std::endl;
//
////        // strerror is NOT thread safe!
////        std::cerr << "Error: " << strerror(errno);
////
////        char buffer[1000];
////        char* answer = getcwd(buffer, sizeof(buffer));
////        std::string s_cwd;
////        if(answer)
////        {
////            s_cwd = answer;
////            std::cout << s_cwd << std::endl;
////        }
//
//
//        exit(1);
//    }
//
//}


//void FileReader::readLIBSVM(SVMProblem& prob)
//{
//    size_t j = 0;
//
//    // Allocate memory for training samples
//    prob.y = new double[prob.l];
//    prob.x = new SVMNode*[prob.l];
//    SVMNode* xSpace = new SVMNode[elements];
//
//    for(int i = 0; i < prob.l; ++i)
//    {
//        prob.x[i] = &xSpace[j];
//        prob.y[i] = atof(trainingSet[i][0].c_str());
//
//        for(size_t n = 1; n < trainingSet[i].size(); ++n)
//        {
//            // Tokenize with : delimiter
//            std::vector<std::string> node = splitString(trainingSet[i][n], ':');
//
//            xSpace[j].index = atoi(node[0].c_str()); // index
//            xSpace[j].value = atof(node[1].c_str()); // Feature value
//
//            //std::cout << "(" << xSpace[j].index << "," << xSpace[j].value << ")" << " ";
//
//            ++j; // next Node
//
//        }
//
//        //std::cout << std::endl;
//        xSpace[j++].index = -1; // last node of i-th sample
//
//    }
//}

// Print CSV data
//void FileReader::printData(SVMProblem& prob)
//{
//
//
//}

//stdvecvec FileReader::convertVecD()
//{
//    stdvecvec dataVec(dataList.size(), stdvec(dataList[0].size()));
//
//    for(unsigned int i = 0; i < dataList.size(); i++)
//    {
//        for(unsigned int j = 0; j < dataList[i].size(); j++)
//        {
//            dataVec[i][j] = atof(dataList[i][j].c_str());
//
//        }
//
//    }
//
//    return dataVec;
//}

//void FileReader::getData(bool ignoreheader)
//{
//    std::ifstream data(fileName.c_str());
//    std::string line;
//
//    // Skip first line
//    if(!ignoreheader == true)
//    {
//        std::getline(data, line);
//    }
//
//    while(std::getline(data, line))
//    {
//        std::stringstream lineStream(line);
//        std::string cell;
//        std::vector<std::string> tempList;
//
//        while(std::getline(lineStream, cell, ','))
//        {
//            tempList.push_back(cell);
//        }
//
//        // Assign tempList to dataList
//        dataList.push_back(tempList);
//
//    }
//
//    data.close();
//
//    // For debugging porpuse...
//    //cout << dataList.size() << endl;
//
//}
