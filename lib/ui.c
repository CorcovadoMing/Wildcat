#include "lib/ui.h"
#include "lib/input.h"
#include "lib/conv.h"
#include "lib/activator.h"
#include "lib/batchnorm.h"
#include "lib/pooling.h"
#include "lib/linear.h"
#include <stdlib.h>

Net* initialize() {
    Net* new = malloc(sizeof(Net));
    new->layer = NULL;
    new->last_layer = NULL;
    new->input = &input;
    new->conv = &conv;
    new->relu = &relu;
    new->maxpooling = &maxpooling;
    new->batchnorm = &batchnorm;
    new->linear = &linear;
    new->softmax = &softmax;
    new->forward = &forward;
    return new;
}
