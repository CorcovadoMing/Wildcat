#include "lib/input.h"
#include <stdarg.h>
#include <stdlib.h>

Net* input(Net* self, int dim, ...) {
    va_list valist;
    va_start(valist, dim);
    Layer* new = layer_constructor("input", 0);
    new->input_dim = dim;
    new->input_shape = malloc(dim * sizeof(int));
    new->output_dim = dim;
    new->output_shape = malloc(dim * sizeof(int));

    for (int i = 0; i < dim; i += 1) {
        new->input_shape[i] = va_arg(valist, int);
        new->output_shape[i] = new->input_shape[i];
    }

    va_end(valist);
    new->_forward = &_input_forward;
    return append(self, new);
}

void _input_forward(Layer* layer, Tensor* data) {
    return;
}
