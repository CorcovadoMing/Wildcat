#include "lib/type.h"
#include "lib/activator.h"
#include <stdlib.h>
#include <math.h>

/*
 * =================================
 * ReLU
 * =================================
 */

Net* relu(Net* self) {
    Layer* new = layer_constructor("relu", 0);
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

    new->_forward = &_relu_forward;
    return append(self, new);
}

void _relu_forward(Layer* layer, Tensor* data) {
    for (int i = 0; i < data->total; i += 1) {
        data->data[i] = data->data[i] < 0 ? 0 : data->data[i];
    }
}


/*
 * =================================
 * Softmax
 * =================================
 */


Net* softmax(Net* self) {
    Layer* new = layer_constructor("softmax", 0);
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

    new->_forward = _softmax_forward;
    return append(self, new);
}

void _softmax_forward(Layer* layer, Tensor* data) {
    for (int i = 0; i < data->shape[0]; i += 1) {
        data_t sum = 0;

        for (int j = 0; j < layer->output_shape[0]; j += 1) {
            sum += exp(at(data, i, j));
        }

        for (int j = 0; j < layer->output_shape[0]; j += 1) {
            assign(exp(at(data, i, j)) / sum,
                   data,
                   i, j);
        }
    }
}
