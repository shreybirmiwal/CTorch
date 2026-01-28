#pragma once

#include "node.h"

void getTopo(struct Value *head, struct Value **topologicalArray, int *size);
void backward(struct Value **topologicalArray, int size);
void freeGraph(struct Value **topologicalArray, int size);
void zeroGrad(struct Value **topologicalArray, int size);

