#include "autograd.h"
#include <time.h>
#include <stdlib.h>

int main(void)
{
    srand(time(NULL));

    struct Value *newVal = createLeafValue(3.0);
    struct Value *val2 = createLeafValue(2.0);
    struct Value *out = addValue(newVal, val2);
    printValue(out);

    // build topologicalArray here
    // a) we need to know how much mem to allocate
    struct Value **topologicalArray = calloc(3, __SIZEOF_POINTER__); // 3 pointers, 3 calues
    int size = 0;
    getTopo(out, topologicalArray, &size);

    // backward pass
    backward(topologicalArray, size);

    printValue(val2);

    // clean up
    zeroGrad(topologicalArray, size);
    freeGraph(topologicalArray, size);
    return 0;
}
