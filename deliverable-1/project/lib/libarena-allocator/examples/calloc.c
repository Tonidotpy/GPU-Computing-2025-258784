/*!
 * \file calloc.c
 * \date 2025-03-20
 * \authors Antonio Gelain [antonio.gelain2@gmail.com]
 *
 * \brief Simple example of a signal convolution with multiple allocations of
 *      homogeneous types using the arena allocator.
 * \details In the example a 3D vector structure consisting of three floats and
 *      another float are allocated and their product is saved in another vector
 *      allocated in the same way.
 *      At the end of the program all the allocated memory is freed to avoid
 *      memory leaks.
 */
#include <stdio.h>
#include <stdlib.h>

#include "arena-allocator.h"
#include "arena-allocator-api.h"

int main(void) {
    ArenaAllocatorHandler_t harena;
    arena_allocator_api_init(&harena);

    /*! Length of the two input signal and the resulting convolution */
    const int N = 10;
    const int M = 5;
    const int C = N + M - 1;

    /*! Initialization of the two signals */
    float *a = (float *)arena_allocator_api_calloc(&harena, N, sizeof(*a));
    float *b = (float *)arena_allocator_api_calloc(&harena, M, sizeof(*b));

    printf("a = [ ");
    for (int i = 0; i < N; ++i) {
        a[i] = (rand() % 1000) / 100.f;
        printf("%.2f ", a[i]);
    }
    printf("]\n");
    printf("b = [ ");
    for (int i = 0; i < M; ++i) {
        b[i] = (rand() % 1000) / 100.f;
        printf("%.2f ", b[i]);
    }
    printf("]\n");

    /*! Initialization and calculation of the convolution */
    float *c = (float *)arena_allocator_api_calloc(&harena, C, sizeof(*c));
    for (int i = 0; i < C; ++i)
        c[i] = 0.f;

    for (int i = 0; i < C; ++i) {
        for (int j = 0; j < M; ++j) {
            int k = i - j;
            if (k >= 0 && k < N) {
                c[i] += a[k] * b[j];
            }
        }
    }

    printf("c = [ ");
    for (int i = 0; i < C; ++i) {
        printf("%.2f ", c[i]);
    }
    printf("]\n");

    /*! Remember to always free the memory at the end of the program */
    arena_allocator_api_free(&harena);
    return 0;
}
