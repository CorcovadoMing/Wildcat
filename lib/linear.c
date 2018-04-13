#include "lib/linear.h"
#include <stdlib.h>
#include <string.h>

Net* linear(Net* self, int out_dimensions) {
    Layer* new = layer_constructor("linear", 1);
    new->arg_list[0] = out_dimensions;
    new->input_dim = self->last_layer->output_dim;
    new->input_shape = malloc(new->input_dim * sizeof(int));
    int total = 1;

    for (int i = 0; i < new->input_dim; i += 1) {
        new->input_shape[i] = self->last_layer->output_shape[i];
        total *= new->input_shape[i];
    }

    // Force reshaping if needed
    new->output_dim = 1;
    new->output_shape = malloc(new->output_dim * sizeof(int));
    new->output_shape[0] = out_dimensions;
    new->weights = vempty_tensor(2, total, out_dimensions);
    new->bias = vempty_tensor(1, out_dimensions);
    new->_forward = &_linear_forward;
    return append(self, new);
}

void _linear_forward(Layer* layer, Tensor* data) {
    Tensor* result = preserve_result(layer, data);
    // although the memory are implemented as a big flat chunk,
    // the shape of tensors limits the layout representation,
    // it could be changed by calling `view`
    view(data, 2, data->shape[0], layer->input_shape[0]);

    for (int i = 0; i < data->shape[0]; i += 1) {
        for (int x = 0; x < layer->weights->shape[0]; x += 1) {
            for (int y = 0; y < layer->weights->shape[1]; y += 1) {
                add(at(data, i, x) * at(layer->weights, x, y),
                    result,
                    i, y);
            }
        }
    }

    swap_and_clear(result, data);
}
