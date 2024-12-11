#include <stdio.h>

int main() {
#ifdef __AVX512BF16__
    printf("AVX-512 BF16 is enabled.\n");
#else
    printf("AVX-512 BF16 is not enabled.\n");
#endif
    return 0;
}
