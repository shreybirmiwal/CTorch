#include <stdbool.h>

struct Value
{
    float data;
    float grad; // this is filled on .backward() later
    bool isLeaf;
    bool isVisited; // for the topological sort

    // if it's not a leaf, we have to fill these in below
    struct Value *child1;
    struct Value *child2;
    char op; // +, *, /, - etc.
};

struct Value *createLeafValue(float data);
void printValue(struct Value *val);
void backward(struct Value *head);

// operations
struct Value *addValue(struct Value *val1, struct Value *val2);
struct Value *multiplyValue(struct Value *val1, struct Value *val2);

// clean up
void freeGraph(struct Value *head);