#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "graph.h"
#include "helper.h"

int main(void)
{

    srand(time(NULL));

    struct Matrix* matrix1 = createRandMatrix(3, 2, true);
    struct Matrix* matrix2 = createRandMatrix(2, 3, true);
    struct Matrix* matrix3 = multiplyMatrix(matrix1, matrix2);
    printf("Matrix 1: \n");
    printMatrix(matrix1);
    printf("Matrix 2: \n");
    printMatrix(matrix2);
    printf("Matrix 3: \n");
    printMatrix(matrix3);

    return 0;
}