#include "autograd.h"
#include <stdlib.h>

struct Tensor *createTensor(int width, bool createRandVals)
{
    struct Tensor *tensor = malloc(sizeof(struct Tensor));
    tensor->width = width;
    tensor->array = calloc(width, sizeof(struct Value *));
    // width amount of pointers in the array

    for (int i = 0; i < width; i++)
    {
        if (createRandVals)
        {
            tensor->array[i] = createRandomLeafValue();
        }
        else
        {
            tensor->array[i] = createLeafValue(0);
        }
    }

    return tensor;
}

struct Embedding_Table *create_embedding_table(int length, int width)
{
    struct Embedding_Table *emb_table = malloc(sizeof(struct Embedding_Table));
    emb_table->size = length;
    for (int i = 0; i < length; i++)
    {
        struct Tensor *row = createTensor(width, true);
        emb_table->table[i] = row;
    }
    return emb_table;
}