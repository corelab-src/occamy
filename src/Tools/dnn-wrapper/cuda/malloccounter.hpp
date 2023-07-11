#include <stdio.h>


static int malloc_counter = 0;

void mallocCounter() {
  printf("[Malloc Counter] %d\n", malloc_counter++);
}

