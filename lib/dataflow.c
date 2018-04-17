#include "lib/dataflow.h"
#include "lib/net.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

DataFlow* empty(int dim, ...) {
    DataFlow* new = malloc(sizeof(DataFlow));
    int* shape = malloc(dim * sizeof(int));
    va_list valist;
    va_start(valist, dim);

    for (int i = 0; i < dim; i += 1) {
        shape[i] = va_arg(valist, int);
    }

    va_end(valist);
    new->data = empty_tensor(dim, shape);
    new->slice = empty_tensor(dim, shape);
    new->result = NULL;
    free(shape);
    // FIXME: should check the dimention first
    new->total = new->data->shape[0];
    new->batch_size = new->total;
    new->cur_iteration = 1;
    new->max_iteration = 1;
    new->batch = &batch;
    return new;
}

DataFlow* batch(DataFlow* self, int batch_size) {
    if (self->total < batch_size) {
        batch_size = self->total;
    }

    self->batch_size = batch_size;
    self->cur_iteration = 1;
    self->max_iteration = (int)ceil(self->total / (float)batch_size);
    self->slice->shape[0] = batch_size;
    self->slice->total = batch_size * (self->slice->total / self->total);
    self->slice->data = malloc(self->slice->total * sizeof(data_t));
    memcpy(self->slice->data, self->data->data, self->slice->total * sizeof(data_t));
    return self;
}

void next_batch(DataFlow* self) {
    self->cur_iteration += 1;
    int batch_size = 0;

    if (self->cur_iteration == self->max_iteration) {
        batch_size = self->total - ((self->cur_iteration - 1) * self->batch_size);
    } else {
        batch_size = self->batch_size;
    }

    free(self->slice->shape);
    free(self->slice->data);
    self->slice->dimention = self->data->dimention;
    self->slice->shape = malloc(self->slice->dimention * sizeof(int));

    for (int i = 0; i < self->data->dimention; i += 1) {
        self->slice->shape[i] = self->data->shape[i];
    }

    self->slice->shape[0] = batch_size;
    long long total = 1;

    for (int i = 0; i < self->slice->dimention; i += 1) {
        total *= self->slice->shape[i];
    }

    self->slice->data = malloc(total * sizeof(data_t));
    int loc = (self->cur_iteration - 1) * self->batch_size;
    memcpy(self->slice->data, self->data->data + loc, self->slice->total * sizeof(data_t));
}

void describe(DataFlow* self) {
    printf("DataFlow Info:\n");
    printf("===========================\n");
    printf("Shape: ");

    for (int i = 0; i < self->data->dimention; i += 1) {
        printf("%d ", self->data->shape[i]);
    }

    printf("\n");
    printf("Batch Size: %d, Current iteration: %d, Max Iteration: %d\n", self->batch_size, self->cur_iteration, self->max_iteration);
    printf("\n");
}
