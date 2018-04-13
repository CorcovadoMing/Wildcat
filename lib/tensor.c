#include "lib/tensor.h"
#include <stdlib.h>
#include <string.h>

Tensor* empty_tensor(int dim, int* shape) {
    Tensor* new = malloc(sizeof(Tensor));
    new->dimention = dim;
    new->shape = malloc(dim * sizeof(int));
    memcpy(new->shape, shape, dim * sizeof(int));
    long long total = 1;

    for (int i = 0; i < dim; i += 1) {
        total *= new->shape[i];
    }

    new->data = calloc(total, sizeof(double));
    new->total = total;
    return new;
}

Tensor* vempty_tensor(int dim, ...) {
    int* shape = malloc(dim * sizeof(int));
    va_list valist;
    va_start(valist, dim);

    for (int i = 0; i < dim; i += 1) {
        shape[i] = va_arg(valist, int);
    }

    va_end(valist);
    Tensor* new = empty_tensor(dim, shape);
    free(shape);
    return new;
}

long long offset(int dim, int* _pos, va_list valist) {
    int* _offset = malloc(dim * sizeof(int));

    for (int i = dim - 1; i >= 0; i -= 1) {
        int size = 1;

        for (int j = i + 1; j < dim; j += 1) {
            size *= _pos[j];
        }

        _offset[i] = size;
    }

    long long loc = 0;

    for (int i = 0; i < dim; i += 1) {
        loc += va_arg(valist, int) * _offset[i];
    }

    return loc;
}

void view(Tensor* self, int dim, ...) {
    //FIXME: check the total number first!
    self->dimention = dim;
    free(self->shape);
    self->shape = malloc(dim * sizeof(int));
    va_list valist;
    va_start(valist, dim);

    for (int i = 0; i < dim; i += 1) {
        self->shape[i] = va_arg(valist, int);
    }
}

double at(Tensor* self, ...) {
    va_list valist;
    va_start(valist, self->dimention);
    long long loc = offset(self->dimention, self->shape, valist);
    va_end(valist);
    return self->data[loc];
}

void assign(double value, Tensor* self, ...) {
    va_list valist;
    va_start(valist, self->dimention);
    long long loc = offset(self->dimention, self->shape, valist);
    va_end(valist);
    self->data[loc] = value;
}

void add(double value, Tensor* self, ...) {
    va_list valist;
    va_start(valist, self->dimention);
    long long loc = offset(self->dimention, self->shape, valist);
    va_end(valist);
    self->data[loc] += value;
}
