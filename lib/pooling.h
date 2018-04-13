#ifndef __MING_WILDCAT_PROJECT_POOLING__
#define __MING_WILDCAT_PROJECT_POOLING__

#include "lib/net.h"

Net* maxpooling(Net*, int, int);
void _maxpooling_forward(Layer*, Tensor*);

#endif
