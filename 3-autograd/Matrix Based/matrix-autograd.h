#include <stdbool.h>
#pragma once

struct Matrix
{
    int height;
    int length;

    float *data;
    float *grads;

    bool isLeaf;
    bool isVisited;   // for topological sort
    bool isUpdatable; // should we update this as a weight / bias? or is it itermediate value

    // If it's not a leaf, we fill these in.
    struct Matrix *child1;
    struct Matrix *child2;
    char op;
};

struct Matrix *createMatrix(int height, int length, float *data, bool isUpdatable);
struct Matrix *createRandMatrix(int height, int length, bool isUpdatable);