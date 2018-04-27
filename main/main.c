#include <stdio.h>
#include <stdlib.h>

//#define USE_INT

#include "lib/ui.h"
#include "lib/net.h"

int main() {
    // Network
    Net* net = initialize();
    net                                  \
    -> input(net, 3, 2, 32, 32)          \
    -> conv(net, 32, 3, 3, 0, 0, 2, 2)   \
    -> batchnorm(net)                    \
    -> relu(net)                         \
    -> conv(net, 32, 3, 3, 0, 0, 2, 2)   \
    -> batchnorm(net)                    \
    -> relu(net)                         \
    -> linear(net, 512);
    summary(net);
    // Data
    DataFlow* data = empty(4, 1, 2, 32, 32); // four-dimension, with (32x32x2) and total 1 image input
    describe(data);
    // Forward
    net->forward(net, data);

    for (int i = 0; i < data->result->shape[0]; i += 1) {
        for (int j = 0; j < data->result->shape[1]; j += 1) {
            printf("%f ", at(data->result, i, j));
        }

        printf("\n");
    }

    // Output shape
    printf("%d %d\n", data->result->shape[0], data->result->shape[1]);

    return 0;
}
