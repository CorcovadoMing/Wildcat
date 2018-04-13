#ifndef __MING_WILDCAT_PROJECT_NET__
#define __MING_WILDCAT_PROJECT_NET__

#include "lib/layer.h"
#include "lib/dataflow.h"

typedef struct _net Net;
struct _net {
    Layer* layer;
    Layer* last_layer;

    Net* (* input)(Net*, int, ...);
    Net* (* conv)(Net*, int, int, int, int, int, int, int);
    Net* (* relu)(Net*);
    Net* (* maxpooling)(Net*, int, int);
    Net* (* batchnorm)(Net*);
    Net* (* linear)(Net*, int);
    Net* (* softmax)(Net*);

    Net* (* forward)(Net*, DataFlow*);
};

void summary(Net*);
Net* append(Net*, Layer*);
Net* forward(Net*, DataFlow*);

#endif
