#ifndef __MING_WILDCAT_PROJECT_ACTIVATOR__
#define __MING_WILDCAT_PROJECT_ACTIVATOR__

#include "lib/net.h"

Net* relu(Net*);
void _relu_forward(Layer*, Tensor*);
Net* softmax(Net*);
void _softmax_forward(Layer*, Tensor*);

#endif
