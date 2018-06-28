#ifndef FILEREADER_H_INCLUDED
#define FILEREADER_H_INCLUDED

#include <string>
#include <vector>

typedef std::vector<std::vector<double> > stdvecvec;
typedef std::vector<double> stdvec;


class FileReader
{
    public:

        std::string fileName;
        std::string delimeter;
        // A vector of vectors to store data
        std::vector <std::vector <std::string> > dataList;
        // This pair holds data(features) and class labels
        //std::pair<mat, imat> dataPair;

        // cotr
        FileReader(std::string filename, std::string delm=","):
            fileName(filename), delimeter(delm) {}

        // Function to fetch data from CSV file
        void getData(bool ignoreheader=false);

        // Print data
        void printData();

        // Convert STL vector to Arma matrix for further operations
        void convertMatrix();

        // Convert STL vector<string> to STL vector<double>
        stdvecvec convertVecD();

        // Convert STL vector<double> to Arma mat
        //mat std_vec_to_mat(stdvecvec &VecVec);



    protected:

    private:
};


#endif // CSVREADER_H_INCLUDED
