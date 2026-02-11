#include "matrix.h"

// void printValue(struct Matrix *val);
// void printTree(struct Matrix *head);

void getTopo(struct Matrix *head, struct Matrix **topologicalArray, int *size);
void backward(struct Matrix **topologicalArray, int size);
void zeroGrad(struct Matrix **topologicalArray, int size);
// void freeGraph(struct Matrix **topologicalArray, int size);
