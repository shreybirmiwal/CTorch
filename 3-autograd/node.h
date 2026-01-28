#pragma once

#include <stdbool.h>

// Core scalar value in the autograd graph.
struct Value
{
    float data;
    float grad; // filled during backward pass
    bool isLeaf;
    bool isVisited; // for topological sort

    // If it's not a leaf, we fill these in.
    struct Value *child1;
    struct Value *child2;
    char op; // '+', '*', etc. '\0' for leaf
};

// Operations / constructors.
struct Value *createLeafValue(float data);
struct Value *addValue(struct Value *val1, struct Value *val2);
struct Value *multiplyValue(struct Value *val1, struct Value *val2);

void printValue(struct Value *val);

