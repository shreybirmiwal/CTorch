#include "graph.h"
#include <stdio.h>
#include <stdlib.h>

void getTopo(struct Matrix *head, struct Matrix **topologicalArray, int *size)
{
    if (head == NULL || head->isVisited == 1)
    {
        return;
    }

    head->isVisited = true;
    // get children first
    getTopo(head->child1, topologicalArray, size);
    getTopo(head->child2, topologicalArray, size);

    // add this one
    topologicalArray[*size] = head;
    *size += 1;
}

// void backward(struct Matrix **topologicalArray, int size)
// {
//     // we have topoligcal array in format: child leaf, child all the way to --> head
//     topologicalArray[size - 1]->grad = 1; // initial gradient has to be set to 1. D-out/D-out = 1

//     for (int i = size - 1; i >= 0; i--)
//     {
//         struct Value *currentNode = topologicalArray[i];

//         switch (currentNode->op)
//         {
//         case 'l': // leaf
//             break;

//         case '+':
//             currentNode->child1->grad += currentNode->grad;
//             currentNode->child2->grad += currentNode->grad;
//             break;
//         case '*':
//             currentNode->child1->grad += (currentNode->child2->data) * currentNode->grad;
//             currentNode->child2->grad += (currentNode->child1->data) * currentNode->grad;
//             break;
//         case 't': // tanh
//             // currentNode->child1->grad += (1 - (tanh(currentNode->child1->data))^2 * * currentNode->grad;
//             //  tanh backward is 1-tanh^2(x)
//             //  we already have tanh(x) from the forward pass, so we can simplify this
//             currentNode->child1->grad += (1 - currentNode->data * currentNode->data) * currentNode->grad;
//             break;

//         default:
//             printf("error! default case hit");
//             break;
//         }
//     }
// }
// void zeroGrad(struct Matrix **topologicalArray, int size)
// {
// }