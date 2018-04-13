#ifndef __MING_WILDCAT_PROJECT_BATCHNORM__
#define __MING_WILDCAT_PROJECT_BATCHNORM__

#include "lib/net.h"

Net* batchnorm(Net*);
void _batchnorm_forward(Layer*, Tensor*);

#endif
