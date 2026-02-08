
struct LinkedListNode
{
    struct LinkedListNode *next;
    struct LinkedListNode *prev;
    int64_t data;
};

typedef struct LinkedListNode LinkedListNode;
// basically saying `struct LinkedListNode` is the same as `LinkedListNode`

typedef struct LinkedListNode *node_pointer;
// basically saying `struct LinkedListNode*` is the same as `node_pointer`
