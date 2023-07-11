#include <stdio.h>


static int memcpy_counter = 0;

void memcpyCounter() {
  printf("[Memcpy Counter] %d\n", memcpy_counter++);
}

