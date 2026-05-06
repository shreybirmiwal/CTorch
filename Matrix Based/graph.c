#include "graph.h"
#include "helper.h"
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

void backward(struct Matrix **topologicalArray, int size)
{
    // we have topoligcal array in format: child leaf, child all the way to --> head
    setEvenGrads(topologicalArray[size - 1], 1); // initial gradient has to be set to 1. D-out/D-out = 1

    for (int i = size - 1; i >= 0; i--)
    {
        struct Matrix *currentNode = topologicalArray[i];

        switch (currentNode->op)
        {
        case 'l': // leaf
            break;

        case '@':{
            //child1
            //struct Matrix* transposed_child2 = transposeMatrix(currentNode->child2);
            float* transposed_child2 = calloc(currentNode->child2->rows * currentNode->child2->cols, sizeof(float));
            backward_helper_transpose(currentNode->child2->data, currentNode->child2->rows, currentNode->child2->cols, transposed_child2);
            backward_helper_matmul(currentNode->grads, currentNode->rows, currentNode->cols, transposed_child2, currentNode->child2->cols, currentNode->child2->rows, currentNode->child1->grads);

            //child2
            // struct Matrix* transposed_child1 = transposeMatrix(currentNode->child1);
            float* transposed_child1 = calloc(currentNode->child1->rows * currentNode->child1->cols, sizeof(float));
            backward_helper_transpose(currentNode->child1->data, currentNode->child1->rows, currentNode->child1->cols, transposed_child1);
            backward_helper_matmul(transposed_child1, currentNode->child1->cols, currentNode->child1->rows, currentNode->grads, currentNode->rows, currentNode->cols, currentNode->child2->grads);
            
            
            free(transposed_child1);
            free(transposed_child2);

            break;
            }

        default:
            printf("error! default case hit");
            break;
        }
    }
}

void zeroGrad(struct Matrix **topologicalArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int g = 0; g < topologicalArray[i]->rows * topologicalArray[i]->cols; g++)
        {
            topologicalArray[i]->grads[g] = 0;
        }
    }
}

void setEvenGrads(struct Matrix *mat, int grad)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->grads[i] = grad;
    }
}



// we want to free the grpah
// but we don't want to loose references, say if we did training in a loop, we would to keep the the same data and gradients? come back to this and think thru it 
void freeGraph(struct Matrix **topologicalArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        //dont we need to go through each child of each node?
        free(topologicalArray[i]->data);
        free(topologicalArray[i]->grads);
        free(topologicalArray[i]);
    }
    
    free(topologicalArray);
}