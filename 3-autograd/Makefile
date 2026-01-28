CC := gcc
CFLAGS := -Wall -Wextra -Wpedantic -std=c11 -O0 -g

TARGET := autograd
SRCS := autograd.c node.c graph.c
OBJS := $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

%.o: %.c autograd.h node.h graph.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
