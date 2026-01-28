#include "autograd.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    struct Value *newVal = createLeafValue(3.0);
    struct Value *val2 = createLeafValue(2.0);
    struct Value *out = addValue(newVal, val2);
    printValue(out);

    struct Value *topographic = getTopo(out);

    // backward pass
    out->grad = 1;
    backward(out);

    printValue(val2);

    // clean up
    freeGraph(out);
    return 0;
}

struct Value *createLeafValue(float data)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));
    newValPointer->data = data;
    newValPointer->grad = 0.0;
    newValPointer->isLeaf = 1;
    newValPointer->child1 = NULL;
    newValPointer->child2 = NULL;
    newValPointer->op = '\0';

    return newValPointer;
}

struct Value *addValue(struct Value *val1, struct Value *val2)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));

    newValPointer->data = val1->data + val2->data;
    newValPointer->grad = 0.0;
    newValPointer->isLeaf = 0;
    newValPointer->child1 = val1;
    newValPointer->child2 = val2;
    newValPointer->op = '+';

    return newValPointer;
}

struct Value *multiplyValue(struct Value *val1, struct Value *val2)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));

    newValPointer->data = (val1->data) * (val2->data);
    newValPointer->grad = 0.0;
    newValPointer->isLeaf = 0;
    newValPointer->child1 = val1;
    newValPointer->child2 = val2;
    newValPointer->op = '*';

    return newValPointer;
}

void getTopo(struct Value *head, struct Value *)
{
    if (head->isVisited == 1)
    {
        return;
    }
}
// sets the children gradients!
void backward(struct Value *head)
{
    // assume head initial gradient is already set to 1
    // assumes each node is only used 1 time

    if (head->op == '\0')
    {
        return;
    }

    else if (head->op == '+')
    {
        // just pass on grad to children
        head->child1->grad += head->grad;
        head->child2->grad += head->grad;

        backward(head->child1);
        backward(head->child2);
    }

    else if (head->op == '*')
    {
        head->child1->grad += (head->child2->data) * head->grad;
        head->child2->grad += (head->child1->data) * head->grad;

        backward(head->child1);
        backward(head->child2);
    }
}

void zeroGrad(struct Value *head)
{
}

void printValue(struct Value *val)
{
    printf("Value\nData: %f\nGrad: %f\nisLeaf: %d\nChild1 Pointer: %p\nChild2 Pointer: %p\nOperation: %c", val->data, val->grad, val->isLeaf, val->child1, val->child2, val->op);
}

void freeGraph(struct Value *head)
{
    if (head->child1 != NULL)
    {
        freeGraph(head->child1);
    }
    if (head->child2 != NULL)
    {
        freeGraph(head->child2);
    }
    free(head);
}
