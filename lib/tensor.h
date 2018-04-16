#ifndef __MING_WILDCAT_PROJECT_TENSOR__
#define __MING_WILDCAT_PROJECT_TENSOR__

#include <stdarg.h>
#include "lib/type.h"

typedef struct _tensor Tensor;
struct _tensor {
    int dimention;
    long long total;
    int* shape;
    data_t* data;
};

Tensor* empty_tensor(int, int*);
Tensor* vempty_tensor(int, ...);
long long offset(int, int*, va_list);
void view(Tensor*, int, ...);
data_t at(Tensor*, ...);
void assign(double, Tensor*, ...);
void add(double, Tensor*, ...);

#endif
