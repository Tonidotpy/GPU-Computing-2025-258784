# LIBARENA-ALLOCATOR

This library implements a simple arena allocator using a dynamic memory array.

## Arena Allocator

An arena allocator is a specific type of dynamic memory allocator which stores
a reference to the allocated items inside a collection, often called ***Arena***,
and can only free all the memory at once.

This type of allocator is one of the most simple and basic types of allocators
since it handles memory management asserting that the allocated memory is always
needed until deallocation, this can greatly reduce fragmentation without losing
too much performance.

The major backdraw of this type of system is that single items can't be deallocated
without removing all the content inside the arena.

> [!TIP]
> Dynamic allocation is not considered a good practice in embedded systems,
> expecially for critical applications, anyway if needed, as suggested
> from important organizations (such as NASA), all the allocations should be
> done once at the start of the program which is what the arena allocator
> can do best.

## Usage

To use this library include the `arena-allocator-api.h` header file inside your
program, declare an arena allocator handler and initialize it with the appropriate
function before any allocation.

> [!IMPORTANT]
> Remeber to free the arena memory at the end of the program, even if it never
> ends.

```c
int main(void) {
    ArenaAllocatorHandler_t harena;
    arena_allocator_api_init(&harena);

    // Do allocations here...

    arena_allocator_api_free(&harena);
}
```

To allocate memory two methods are available:
1. Allocation of a single element
2. Allocation of multiple homogeneous elements (an array basically)

> [!NOTE]
> Different allocations are not needed to use the same variable types, the
> allocator is type agnostic and is based solely on the type size in bytes.

For more info check the [examples](examples) folder.
