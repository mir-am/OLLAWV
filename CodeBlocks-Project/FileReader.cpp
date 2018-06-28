#include "FILEReader.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>


//CSVReader::CSVReader()
//{
//    //ctor
//}

void FileReader::getData(bool ignoreheader)
{
    std::ifstream data(fileName.c_str());
    std::string line;

    // Skip first line
    if(!ignoreheader == true)
    {
        std::getline(data, line);
    }

    while(std::getline(data, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> tempList;

        while(std::getline(lineStream, cell, ','))
        {
            tempList.push_back(cell);
        }

        // Assign tempList to dataList
        dataList.push_back(tempList);

    }

    data.close();

    // For debugging porpuse...
    //cout << dataList.size() << endl;

}


// Print CSV data
void FileReader::printData()
{
    for(unsigned int i=0; i < dataList.size(); i++)
    {
        for(unsigned int j=0; j < dataList[i].size(); j++)
        {
            std::cout << dataList[i][j] << " ";
        }

        std::cout << "\n";
    }

}


//void CSVReader::convertMatrix()
//{
//
//    // Create a matrix for data(features)
//    mat dataMatrix(dataList.size(), dataList[0].size() - 1, fill::zeros);
//
//    // Create a matrix for storing class labels
//    imat labelMatrix(dataList.size(), 1, fill::zeros);
//
//    for(unsigned int i=0; i < dataList.size(); i++)
//    {
//        labelMatrix(i, 0) = atoi(dataList[i][0].c_str()); // Class label
//
//        for(unsigned int j=0; j < dataList[i].size() - 1; j++)
//        {
//            dataMatrix(i, j) = atof(dataList[i][j + 1].c_str()); // Features
//
//        }
//    }
//
//    dataPair.first = dataMatrix;
//    dataPair.second = labelMatrix;
//}


stdvecvec FileReader::convertVecD()
{
    stdvecvec dataVec(dataList.size(), stdvec(dataList[0].size()));

    for(unsigned int i = 0; i < dataList.size(); i++)
    {
        for(unsigned int j = 0; j < dataList[i].size(); j++)
        {
            dataVec[i][j] = atof(dataList[i][j].c_str());

        }

    }

    return dataVec;
}


//mat CSVReader::std_vec_to_mat(stdvecvec &VecVec)
//{
//    mat data = zeros<mat>(VecVec.size(), VecVec[0].size());
//
//    for(unsigned int i = 0; i < data.n_rows; i++)
//    {
//        data.row(i) = conv_to<rowvec>::from(VecVec[i]);
//
//    }
//
//    return data;
//}

