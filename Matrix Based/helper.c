#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

void matmul_backward(float* a, int ar, int ac, float* b, int br, int bc, float* c)
{
    

    for (int i = 0; i < ar; i++)
    {
        for (int g = 0; g < bc; g++)
        {

            for (int q = 0; q < ac; q++)
            {
                c[i * bc + g] += a[i*ac+q] * b[q*bc+g]; 

            }

        }
    }
}