#include "lib/batchnorm.h"
#include <stdlib.h>

Net* batchnorm(Net* self) {
    Layer* new = layer_constructor("batchnorm", 0);
    new->input_dim = self->last_layer->output_dim;
    new->input_shape = malloc(new->input_dim * sizeof(int));

    for (int i = 0; i < new->input_dim; i += 1) {
        new->input_shape[i] = self->last_layer->output_shape[i];
    }

    new->output_dim = new->input_dim;
    new->output_shape = malloc(new->output_dim * sizeof(int));

    for (int i = 0; i < new->output_dim; i += 1) {
        new->output_shape[i] = new->input_shape[i];
    }

    new->_forward = &_batchnorm_forward;
    return append(self, new);
}

void _batchnorm_forward(Layer* layer, Tensor* data) {
    return;
}
