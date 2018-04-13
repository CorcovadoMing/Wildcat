#ifndef __MING_WILDCAT_PROJECT_TENSOR__
#define __MING_WILDCAT_PROJECT_TENSOR__

#include <stdarg.h>

typedef struct _tensor Tensor;
struct _tensor {
    int dimention;
    long long total;
    int* shape;
    double* data;
};

Tensor* empty_tensor(int, int*);
Tensor* vempty_tensor(int, ...);
long long offset(int, int*, va_list);
void view(Tensor*, int, ...);
double at(Tensor*, ...);
void assign(double, Tensor*, ...);
void add(double, Tensor*, ...);

#endif
