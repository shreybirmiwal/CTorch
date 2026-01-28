#include "node.h"
#include <stdio.h>
#include <stdlib.h>

struct Value *createLeafValue(float data)
{
    struct Value *newValPointer = malloc(sizeof(struct Value));
    newValPointer->data = data;
    newValPointer->isVisited = 0;
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
    newValPointer->isVisited = 0;
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
    newValPointer->isVisited = 0;
    newValPointer->isLeaf = 0;
    newValPointer->child1 = val1;
    newValPointer->child2 = val2;
    newValPointer->op = '*';

    return newValPointer;
}

void printValue(struct Value *val)
{
    printf("\nValue\nData: %f\nGrad: %f\nisLeaf: %d\nChild1 Pointer: %p\nChild2 Pointer: %p\nOperation: %c",
           val->data,
           val->grad,
           val->isLeaf,
           (void *)val->child1,
           (void *)val->child2,
           val->op);
}
