#include "lib/conv.h"
#include <stdlib.h>

Net* conv(Net* self, int k, int w, int h, int pad_w, int pad_h, int stride_w, int stride_h) {
    Layer* new = layer_constructor("conv", 7);
    new->arg_list[0] = k;
    new->arg_list[1] = w;
    new->arg_list[2] = h;
    new->arg_list[3] = pad_w;
    new->arg_list[4] = pad_h;
    new->arg_list[5] = stride_w;
    new->arg_list[6] = stride_h;
    new->input_dim = self->last_layer->output_dim;
    new->input_shape = malloc(new->input_dim * sizeof(int));

    for (int i = 0; i < new->input_dim; i += 1) {
        new->input_shape[i] = self->last_layer->output_shape[i];
    }

    new->output_dim = new->input_dim;
    new->output_shape = malloc(new->output_dim * sizeof(int));
    new->output_shape[0] = k;
    // ref: http://cs231n.github.io/convolutional-networks/
    new->output_shape[1] = (new->input_shape[1] - w + (2 * pad_w)) / stride_w + 1;
    new->output_shape[2] = (new->input_shape[2] - h + (2 * pad_h)) / stride_h + 1;
    new->weights = vempty_tensor(4, new->input_shape[0], w, h, k);
    new->bias = vempty_tensor(1, k);
    new->_forward = &_conv_forward;
    return append(self, new);
}

void _conv_forward(Layer* layer, Tensor* data) {
    Tensor* result = preserve_result(layer, data);

    // i is batch index, if only 1 sample is inferenced, the batch outer loop could be ignored
    for (int i = 0; i < data->shape[0]; i += 1) {
        for (int out_channel = 0; out_channel < layer->arg_list[0]; out_channel += 1) {
            for (int in_channel = 0; in_channel < layer->input_shape[0]; in_channel += 1) {
                for (int x = 0; x < layer->output_shape[1]; x += 1) {
                    for (int y = 0; y < layer->output_shape[2]; y += 1) {
                        double data_pixel = 0;
                        double weight_pixel = 0;

                        for (int w = 0; w < layer->arg_list[1]; w += 1) {
                            for (int h = 0; h < layer->arg_list[2]; h += 1) {
                                // Padding is not implmented yet,
                                // The implement idea is to modify the x, y range,
                                // Then check while the range is out of Tensor shape,
                                // data_pixel becomes 0, as zero-padding does
                                data_pixel = at(data, i, in_channel, (x * layer->arg_list[5]) + w, (y * layer->arg_list[6]) + h);
                                weight_pixel = at(layer->weights, in_channel, w, h, out_channel);
                                add(data_pixel * weight_pixel,
                                    result,
                                    i, out_channel, x, y);
                                // bias
                                add(at(layer->bias, out_channel),
                                    result,
                                    i, out_channel, x, y);
                            }
                        }
                    }
                }
            }
        }
    }

    swap_and_clear(result, data);
}
