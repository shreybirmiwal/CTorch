#include "graph.h"
#include <stdio.h>
#include <stdlib.h>

//  struct Value **topologicalArray is a pointer to a POIINTER OF STRUCT VALUES
// this means we are pointing to where we have an array of struct value pointers
void getTopo(struct Value *head, struct Value **topologicalArray, int *size)
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
// sets the children gradients!
void backward(struct Value **topologicalArray, int size)
{
    // we have topoligcal array in format: child leaf, child all the way to --> head
    topologicalArray[size - 1]->grad = 1; // initial gradient has to be set to 1. D-out/D-out = 1

    for (int i = size - 1; i >= 0; i--)
    {

        struct Value *currentNode = topologicalArray[i];

        switch (currentNode->op)
        {
        case '\0':
            break;

        case '+':
            currentNode->child1->grad += currentNode->grad;
            currentNode->child2->grad += currentNode->grad;
            break;
        case '*':
            currentNode->child1->grad += (currentNode->child2->data) * currentNode->grad;
            currentNode->child2->grad += (currentNode->child1->data) * currentNode->grad;
            break;

        default:
            break;
        }
    }
}

void freeGraph(struct Value **topologicalArray, int size)
{
    // we want to free the head LAST because then we loose referneces.
    for (int i = 0; i < size; i++)
    {
        free(topologicalArray[i]); // free expects a pointer. since we have double pointer, [] dereferences 1 pointer so we doing right.
    }

    free(topologicalArray);
}

void zeroGrad(struct Value **topologicalArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        topologicalArray[i]->grad = 0;
    }
}
