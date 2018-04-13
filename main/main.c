#include <stdio.h>
#include <stdlib.h>
#include "lib/net.h"
#include "lib/ui.h"

int main() {
    // Network
    Net* net = initialize();
    net                                  \
    -> input(net, 3, 1, 32, 32)          \
    -> conv(net, 32, 3, 3)               \
    -> batchnorm(net)                    \
    -> relu(net)                         \
    -> maxpooling(net, 2, 2)             \
    -> conv(net, 32, 3, 3)               \
    -> batchnorm(net)                    \
    -> relu(net)                         \
    -> linear(net, 10)                   \
    -> softmax(net);
    summary(net);
    // Data
    DataFlow* data = empty(4, 2, 1, 32, 32);
    data->batch(data, 1); // if needed
    describe(data);
    // Forward
    net->forward(net, data);

    for (int i = 0; i < data->result->shape[0]; i += 1) {
        for (int j = 0; j < data->result->shape[1]; j += 1) {
            printf("%f ", at(data->result, i, j));
        }

        printf("\n");
    }

    return 0;
}
