/*!
 * \file arena-allocator-api.h
 * \date 2025-03-01
 * \authors Antonio Gelain [antonio.gelain2@gmail.com]
 *
 * \brief Simple implementation of an arena allocator.
 * \details An arena allocator is a memory management technuque where a large
 *     block of memory (the arena) and uses it to manage smaller allocations
 *     efficiently.
 *     This method allows for faster allocation and deallocation since all
 *     memory can be freed at once when the arena is no longer needed.
 *     This implementation of the allocator uses a dynamic array which gets
 *     reallocated when more memory is needed to keep track of all the
 *     allocated memory.
 *
 * \warning Memory handled by the arena allocator should not be freed outside
 *      of the allocator itself.
 *      This might caused undefined behavior caused by double free of the same
 *      memory region.
 */
#ifndef ARENA_ALLOCATOR_API_H
#define ARENA_ALLOCATOR_API_H

#include "arena-allocator.h"

/*!
 * \brief Initialize the arena allocator handler.
 *
 * \attention This function should always be called before every other function
 *      defined by this library.
 *
 * \param[in] harena A reference to the arena allocator handler.
 */
void arena_allocator_api_init(ArenaAllocatorHandler_t *harena);

/*!
 * \brief Allocate a single item using the arena allocator.
 *
 * \attention The returned reference should be casted to the appropriate type.
 *
 * \param[in] harena A reference to the arena allocator handler.
 * \param[in] size   The amount of memory to allocate in **bytes**.
 * \return A pointer to the allocated memory region or NULL on failure.
 */
void *arena_allocator_api_alloc(ArenaAllocatorHandler_t *harena, size_t size);

/*!
 * \brief Allocate multiple items at once using the arena allocator.
 *
 * \attention The returned reference should be casted to the appropriate type.
 *
 * \param[in] harena A reference to the arena allocator handler.
 * \param[in] size   The amount of memory to allocate for a single item in **bytes**.
 * \param[in] count  The number of items to allocate.
 * \return A pointer to the allocated memory region or NULL on failure.
 */
void *arena_allocator_api_calloc(ArenaAllocatorHandler_t *harena, size_t size, size_t count);

/*!
 * \brief Free all the memory previously allocated by the arena allocator.
 *
 * \attention This function should always be called at the end of the program
 *      to correctly free all the previously allocated memory to avoid incurring
 *      in memory leaks.
 *
 * \param[in] harena A reference to the arena allocator handler.
 */
void arena_allocator_api_free(ArenaAllocatorHandler_t *harena);

#endif // ARENA_ALLOCATOR_API_H
