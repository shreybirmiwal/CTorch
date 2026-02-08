#include <stdio.h>
#include "linked-list.h"
#include <stdint.h>

int main(void)
{
    printf("Hello, world!\n");

    LinkedListNode node;
    node.data = 10;

    printf("Node data: %ld\n", node.data);

    return 0;
}