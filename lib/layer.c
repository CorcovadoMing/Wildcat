#include "lib/layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void show(Layer* layer) {
    printf("Type: %10s, ", layer->type);
    printf("Parameters: ");

    if (layer->weights) {
        printf("%8lld", layer->weights->total);
    } else {
        printf("%8d", 0);
    }

    printf(", Input Shape: ");

    for (int i = 0; i < layer->input_dim; i += 1) {
        printf("%3d ", layer->input_shape[i]);
    }

    printf(", Output Shape: ");

    for (int i = 0; i < layer->output_dim; i += 1) {
        printf("%3d ", layer->output_shape[i]);
    }

    printf("\n");
}

Layer* layer_constructor(char* type, int arg_length) {
    Layer* new = malloc(sizeof(Layer));
    strcpy(new->type, type);
    new->next = NULL;
    new->weights = NULL;
    new->bias = NULL;
    new->_forward = NULL;
    new->arg_length = arg_length;

    if (arg_length) {
        new->arg_list = malloc(arg_length * sizeof(int));
    }

    return new;
}

Tensor* preserve_result(Layer* layer, Tensor* data) {
    int* output_shape = malloc((layer->output_dim + 1) * sizeof(int));
    output_shape[0] = data->shape[0];
    memcpy(output_shape + 1, layer->output_shape, layer->output_dim * sizeof(int));
    Tensor* result = empty_tensor(layer->output_dim + 1, output_shape);
    free(output_shape);
    return result;
}

void swap_and_clear(Tensor* result, Tensor* data) {
    data->dimention = result->dimention;
    data->shape = result->shape;
    data->total = result->total;
    free(data->data);
    data->data = result->data;
    free(result);
}
