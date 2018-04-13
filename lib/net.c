#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include "lib/net.h"

void summary(Net* self) {
    printf("Network Structure:\n");
    printf("===========================\n");
    Layer* layer = self->layer;

    if (layer) {
        show(layer);

        while (layer->next != NULL) {
            layer = layer->next;
            show(layer);
        }
    }

    printf("\n");
}

Net* append(Net* self, Layer* new) {
    if (self->layer == NULL) {
        self->layer = new;
    } else {
        self->last_layer->next = new;
    }

    self->last_layer = new;
    return self;
}

Net* forward(Net* net, DataFlow* data) {
    int result_offset = 1;

    while (data->cur_iteration <= data->max_iteration) {
        Layer* layer = net->layer;
        // printf("Current iter: %d, Max: %d\n", data->cur_iteration, data->max_iteration);

        if (layer) {
            layer->_forward(layer, data->slice);

            while (layer->next != NULL) {
                layer = layer->next;
                layer->_forward(layer, data->slice);
            }
        } else {
            printf("[Error] The network is not defined. Ignored.");
        }

        if (data->result == NULL) {
            int* shape = malloc(data->slice->dimention * sizeof(int));
            shape[0] = data->total;

            for (int i = 1; i < data->slice->dimention; i += 1) {
                shape[i] = data->slice->shape[i];
                result_offset *= shape[i];
            }

            data->result = empty_tensor(data->slice->dimention, shape);
            free(shape);
        }

        memcpy(data->result->data + ((data->cur_iteration - 1) * data->batch_size * result_offset),
               data->slice->data,
               data->slice->total * sizeof(double));
        next_batch(data);
    }

    return net;
}
