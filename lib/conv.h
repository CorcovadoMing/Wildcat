#ifndef __MING_WILDCAT_PROJECT_CONV__
#define __MING_WILDCAT_PROJECT_CONV__

#include "lib/net.h"

Net* conv(Net*, int, int, int);
void _conv_forward(Layer*, Tensor*);

#endif
