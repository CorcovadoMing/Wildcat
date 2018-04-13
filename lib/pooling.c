#include "lib/pooling.h"
#include <stdlib.h>
#include <math.h>

Net* maxpooling(Net* self, int stride_w, int stride_h) {
    Layer* new = layer_constructor("maxpooling", 2);
    new->arg_list[0] = stride_w;
    new->arg_list[1] = stride_h;
    new->input_dim = self->last_layer->output_dim;
    new->input_shape = malloc(new->input_dim * sizeof(int));

    for (int i = 0; i < new->input_dim; i += 1) {
        new->input_shape[i] = self->last_layer->output_shape[i];
    }

    new->output_dim = new->input_dim;
    new->output_shape = malloc(new->output_dim * sizeof(int));
    new->output_shape[0] = new->input_shape[0];
    new->output_shape[1] = new->input_shape[1] / stride_w;
    new->output_shape[2] = new->input_shape[2] / stride_h;
    new->_forward = &_maxpooling_forward;
    return append(self, new);
}

void _maxpooling_forward(Layer* layer, Tensor* data) {
    Tensor* result = preserve_result(layer, data);

    for (int i = 0; i < data->shape[0]; i += 1) {
        for (int in_channel = 0; in_channel < layer->input_shape[0]; in_channel += 1) {
            for (int x = 0; x < layer->output_shape[1]; x += 1) {
                for (int y = 0; y < layer->output_shape[2]; y += 1) {
                    double tmp = at(data, i, in_channel, x, y);

                    for (int comp_x = 0; comp_x < layer->arg_list[0]; comp_x += 1) {
                        for (int comp_y = 0; comp_y < layer->arg_list[1]; comp_y += 1) {
                            tmp = fmax(tmp, at(data, i, in_channel, (layer->arg_list[0] * x) + comp_x, (layer->arg_list[1] * y) + comp_y));
                        }
                    }

                    assign(tmp, result, i, in_channel, x, y);
                }
            }
        }
    }

    swap_and_clear(result, data);
}
