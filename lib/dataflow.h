#ifndef __MING_WILDCAT_PROJECT_DATAFLOW__
#define __MING_WILDCAT_PROJECT_DATAFLOW__

#include "lib/tensor.h"

typedef struct _dataflow DataFlow;
struct _dataflow {
    int total;
    int batch_size;
    int cur_iteration;
    int max_iteration;
    Tensor* data;
    Tensor* slice;
    Tensor* result;
    DataFlow* (* batch)(DataFlow*, int);
};

DataFlow* empty(int, ...);
DataFlow* import(char*);
DataFlow* batch(DataFlow*, int);
void next_batch(DataFlow*);
void describe(DataFlow*);

#endif
