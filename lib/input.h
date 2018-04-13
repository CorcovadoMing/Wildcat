#ifndef __MING_WILDCAT_PROJECT_INPUT__
#define __MING_WILDCAT_PROJECT_INPUT__

#include "lib/net.h"

Net* input(Net*, int, ...);
void _input_forward(Layer*, Tensor*);

#endif
