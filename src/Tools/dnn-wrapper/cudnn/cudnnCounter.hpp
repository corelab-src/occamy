#include <stdio.h>

static int conv_counter = 0;

void convCounter() {
  printf("[Conv COunter] conv %d\n", conv_counter++);
}
