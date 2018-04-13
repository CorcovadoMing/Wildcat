#ifndef __MING_WILDCAT_PROJECT_LAYER__
#define __MING_WILDCAT_PROJECT_LAYER__

#include "lib/tensor.h"

typedef struct _layer Layer;
struct _layer {
    char type[16];
    int arg_length;
    int input_dim;
    int output_dim;
    int* arg_list;
    // CWH format
    int* input_shape;
    int* output_shape;
    Tensor* weights;
    Tensor* bias;
    Layer* next;
    void (* _forward)(Layer*, Tensor*);
};

void show(Layer*);
Layer* layer_constructor(char*, int);
Tensor* preserve_result(Layer*, Tensor*);
void swap_and_clear(Tensor*, Tensor*);

#endif
