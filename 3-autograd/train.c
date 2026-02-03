#include "autograd.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    srand(time(NULL));
    // training y = 3x + 5 for [0-9] for x
    int x = 1; // in
    int y = 8; // out

    // layer 1: embedding table size size 10x9
    struct Embedding_Table *emb_table = create_embedding_table(10, 9);
    struct Tensor *curTensor = emb_table->table[x]; // plug in x
    // layer 2:
    struct Neuron *l2Neuron1 = createNeuron(3);
    struct Value *l2_post_n1 = forwardNeuron(l2Neuron1, get_sub_array_from_tensor(curTensor, 0, 3));
    struct Neuron *l2Neuron2 = createNeuron(3);
    struct Value *l2_post_n2 = forwardNeuron(l2Neuron2, get_sub_array_from_tensor(curTensor, 3, 6));
    // layer 3 tanH on neuron2:
    struct Value *l3_post_n2 = tanhValue(l2_post_n1);
    // layer 2 on neuron3
    struct Neuron *l2Neuron3 = createNeuron(3);
    struct Value *l2_post_n3 = forwardNeuron(l2Neuron2, get_sub_array_from_tensor(curTensor, 6, 9));
    // layer 4
    struct Neuron *l3Neuron = createNeuron(3);
    struct Value **arr_from_layer3 = calloc(3, sizeof(struct Value *));
    arr_from_layer3[0] = l2_post_n1;
    arr_from_layer3[1] = l3_post_n2;
    arr_from_layer3[2] = l2_post_n3;
    struct Value *out = forwardNeuron(l3Neuron, arr_from_layer3);
    // loss MSE
    struct Value *actualVal = createLeafValue(-1 * y, false);
    struct Value *loss = addValue(out, actualVal);
    struct Value *squared_loss = multiplyValue(loss, loss);

    // build topologicalArray here
    struct Value **topologicalArray = calloc(1000, sizeof(struct Value *)); // TODO: we need to know how much mem to allocate, fix
    int size = 0;
    getTopo(squared_loss, topologicalArray, &size);

    // backward pass
    backward(topologicalArray, size);
    printTree(squared_loss);

    // update weights using gradients
    updateWeights(topologicalArray, size, .01);

    // clean up, reset gradients and free graph
    zeroGrad(topologicalArray, size);
    freeGraph(topologicalArray, size);
    return 0;
}

// todo:
// 1. fix topological array line 41
// 2. singular train (post on x)
// 3. train loop + accuracy (post on x)

// 4. fix memory issues (post on x)

// 5. after this training loop remove notion of neurons switch to matrix multiplication
// 6. train MLP as per bengio et all

// 7. train RNN
// 8. train CNN

// resume add: Autograd engine complete, create MLP, RNN, CNN using it