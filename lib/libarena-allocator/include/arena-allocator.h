/*!
 * \file arena-allocator.h
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
#ifndef ARENA_ALLOCATOR_H
#define ARENA_ALLOCATOR_H

#include <stddef.h>

/*!
 * \brief Single item allocated inside the arena.
 * \details Can be any type (even arrays or structures), the actual type of the
 *      item depends on the user and is not tracked by the allocator itself.
 *
 * \note The memory allocated for the actual item data is not contained inside
 *      the allocator, only a reference is kept.
 *
 * \struct ArenaAllocatorItem_t
 * \var void * value
 *      A pointer to the allocated memory of the item.
 */
typedef struct _ArenaAllocatorItem_t {
    void *value;
} ArenaAllocatorItem_t;

/*!
 * \brief Handler structure of the arena allocator.
 * \details The handler contains all the information used to manage a dynamic
 *      array where the items are stored.
 *
 * \struct ArenaAllocatorHandler_t
 * \var size_t size
 *      The total number of items currently stored inside the arena.
 * \var size_t capacity
 *      The maximium amount of items which can be stored inside the arena
 *      without allocation.
 * \var ArenaAllocatorItem_t * items
 *      A pointer to the allocated array where all the arena items are stored.
 */
typedef struct _ArenaAllocatorHandler_t {
    size_t size, capacity;
    ArenaAllocatorItem_t *items;
} ArenaAllocatorHandler_t;

#endif // ARENA_ALLOCATOR_H
