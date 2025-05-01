/*!
 * \file alloc.c
 * \date 2025-03-17
 * \authors Antonio Gelain [antonio.gelain2@gmail.com]
 *
 * \brief Simple example of a vector-scalar multiplication with single
 *      allocations of different types using the arena allocator.
 * \details In the example a 3D vector structure consisting of three floats and
 *      another float are allocated and their product is saved in another vector
 *      allocated in the same way.
 *      At the end of the program all the allocated memory is freed to avoid
 *      memory leaks.
 */
#include <stdio.h>

#include "arena-allocator.h"
#include "arena-allocator-api.h"

/*!
 * \brief 3D vector structure
 *
 * \struct Vec3_t
 * \var float x
 *      The x component of the vector
 * \var float y
 *      The y component of the vector
 * \var float z
 *      The z component of the vector
 */
typedef struct _Vec3_t {
    float x, y, z;
} Vec3_t;

/*!
 * \brief Utility function to print the content of a 3d vector
 *
 * \param v[in] The vector to print
 */
void print_vec3(Vec3_t *v) {
    printf("[%4.2f %4.2f %4.2f]", v->x, v->y, v->z);
}

int main(void) {
    ArenaAllocatorHandler_t harena;
    arena_allocator_api_init(&harena);

    Vec3_t *v = (Vec3_t *)arena_allocator_api_alloc(&harena, sizeof(*v));
    float *k = (float *)arena_allocator_api_alloc(&harena, sizeof(*k));

    /*! Initialization of the vector and scalar values */
    v->x = 1.53f;
    v->y = 5.92f;
    v->z = 2.01f;
    *k = 2.5f;

    printf("%.2f * ", *k);
    print_vec3(v);
    printf(" = ");

    /*! Allocate and calculate a vector multiplication with a scalar value */
    Vec3_t *w = (Vec3_t *)arena_allocator_api_alloc(&harena, sizeof(*w));
    w->x = v->x * (*k);
    w->y = v->y * (*k);
    w->z = v->z * (*k);
    print_vec3(w);
    printf("\n");

    /*! Remember to always free the memory at the end of the program */
    arena_allocator_api_free(&harena);
    return 0;
}
