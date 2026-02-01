#include "autograd.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    srand(time(NULL));

    // training y = 3x + 5 for [0-9] for x

    // layer 1: embedding table size size 10x9
    struct Embedding_Table *emb_table = create_embedding_table(10, 9);
    // plug in x = 0
    struct Tensor *curTensor = emb_table->table[0];

    // layer 2:
    struct Neuron *l2Neuron1 = createNeuron(3);
    struct Value *l2_post_n1 = forwardNeuron(l2Neuron1, curTensor->array[0]);

    struct Neuron *l2Neuron2 = createNeuron(3);
    struct Value *l2_post_n2 = forwardNeuron(l2Neuron2, curTensor->array[3]);
    // layer 3 tanH on neuron2:
    struct Value *l3_post_n2 = tanhValue(l2_post_n1);

    struct Neuron *l2Neuron3 = createNeuron(3);
    struct Value *l2_post_n2 = forwardNeuron(l2Neuron2, curTensor->array[6]);

    // layer 4
    struct Neuron *l3Neuron = createNeuron(3);
    struct Value **arr_from_layer3 = calloc(3, sizeof(struct Value *));
    arr_from_layer3[0] = l2_post_n1;
    arr_from_layer3[1] = l3_post_n2;
    arr_from_layer3[2] = l2_post_n2;
    struct Value *out = forwardNeuron(l3Neuron, arr_from_layer3);

    printValue(out);

    // build topologicalArray here
    // a) we need to know how much mem to allocate
    struct Value **topologicalArray = calloc(3, __SIZEOF_POINTER__); // 3 pointers, 3 calues
    int size = 0;
    getTopo(out, topologicalArray, &size);

    // backward pass
    backward(topologicalArray, size);

    // clean up
    zeroGrad(topologicalArray, size);
    freeGraph(topologicalArray, size);
    return 0;
}
