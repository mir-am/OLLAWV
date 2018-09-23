#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef float Qfloat;

class QMatrix
{
    public:

        // Getting one column from Q matrix
        virtual Qfloat* get_Q(int column, int len) const = 0;
        virtual double* get_QD() const = 0;
        virtual void swap_index(int i, j) const = 0;
        virtual ~QMatrix() {}


};

#endif // OPTIMIZER_H
