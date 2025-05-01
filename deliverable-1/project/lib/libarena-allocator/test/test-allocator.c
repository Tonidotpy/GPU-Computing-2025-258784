#include "unity.h"

#include "arena-allocator-api.h"

#include <malloc.h>

/*! Function declaration needed to test the source only functions */
void *_arena_allocator_api_item_push(ArenaAllocatorHandler_t *harena, size_t size);
void *_arena_allocator_api_item_push_with_alloc(ArenaAllocatorHandler_t *harena, size_t size);

ArenaAllocatorHandler_t harena;

void setUp(void) {
    arena_allocator_api_init(&harena);
}

void tearDown(void) {
    arena_allocator_api_free(&harena);
}

/*!
 * \defgroup item_push Test item push without allocation
 * @{
 */

void test_arena_allocator_api_item_push_address(void) {
    const size_t size = sizeof(int);
    harena.items = malloc(size);
    harena.capacity = 1U;
    void *item = _arena_allocator_api_item_push(&harena, size);
    TEST_ASSERT_EQUAL_PTR(harena.items[0U].value, item);
}

void test_arena_allocator_api_item_push_arena_size(void) {
    const size_t size = sizeof(int);
    harena.items = malloc(size);
    harena.capacity = 1U;
    (void)_arena_allocator_api_item_push(&harena, size);
    TEST_ASSERT_EQUAL_size_t(1U, harena.size);
}

/*! @} */

/*!
 * \defgroup item_push_with_alloc Test item push with allocation
 * @{
 */

void test_arena_allocator_api_item_push_with_alloc_when_zero_capacity_arena_address(void) {
    const size_t size = sizeof(int);
    (void)_arena_allocator_api_item_push_with_alloc(&harena, size);
    TEST_ASSERT_NOT_NULL(harena.items);
}

void test_arena_allocator_api_item_push_with_alloc_when_zero_capacity_capacity(void) {
    const size_t size = sizeof(int);
    (void)_arena_allocator_api_item_push_with_alloc(&harena, size);
    TEST_ASSERT_EQUAL_size_t(1U, harena.capacity);
}

void test_arena_allocator_api_item_push_with_alloc_when_full_arena_address(void) {
    const size_t size = sizeof(int);
    /*! Test with power of two since the arena capacity doubles whenever it is full */
    const size_t n = 1U << 1U;
    for (size_t i = 0U; i < n; ++i)
        (void)_arena_allocator_api_item_push_with_alloc(&harena, size);

    (void)_arena_allocator_api_item_push_with_alloc(&harena, size);
    TEST_ASSERT_NOT_NULL(harena.items);
}

void test_arena_allocator_api_item_push_with_alloc_when_full_capacity(void) {
    const size_t size = sizeof(int);
    /*! Test with power of two since the arena capacity doubles whenever it is full */
    const size_t n = 1U << 1U;
    for (size_t i = 0U; i < n; ++i)
        (void)_arena_allocator_api_item_push_with_alloc(&harena, size);

    (void)_arena_allocator_api_item_push_with_alloc(&harena, size);
    TEST_ASSERT_EQUAL_size_t(n * 2U, harena.capacity);
}

void test_arena_allocator_api_item_push_with_alloc_without_realloc_address(void) {
    const size_t size = sizeof(int);
    /*! Test with 3 items pushed so that the capacity is greater */
    const size_t n = 3U;
    for (size_t i = 0U; i < n; ++i)
        (void)_arena_allocator_api_item_push_with_alloc(&harena, size);

    void *expected = harena.items;
    (void)_arena_allocator_api_item_push_with_alloc(&harena, size);
    TEST_ASSERT_EQUAL_PTR(expected, harena.items);
}

void test_arena_allocator_api_item_push_with_alloc_without_realloc_arena_size(void) {
    const size_t size = sizeof(int);
    /*! Test with 3 items pushed so that the capacity is greater */
    const size_t n = 3U;
    for (size_t i = 0U; i < n; ++i)
        (void)_arena_allocator_api_item_push_with_alloc(&harena, size);

    (void)_arena_allocator_api_item_push_with_alloc(&harena, size);
    TEST_ASSERT_EQUAL_size_t(n + 1, harena.size);
}

/*! @} */

/*!
 * \defgroup init Test initialization
 * @{
 */

void test_arena_allocator_api_init_with_null(void) {
    arena_allocator_api_init(NULL);
    /*!
     * The assertion is always true since, if not handled correctly, the init
     * function should crash given NULL as parameter even if it is not
     * garanteed to.
     */
    TEST_ASSERT_TRUE(1);
}

void test_arena_allocator_api_init_values(void) {
    ArenaAllocatorHandler_t expected = { 0 };
    TEST_ASSERT_EQUAL_MEMORY(&expected, &harena, sizeof(expected));
}

/*! @} */

/*!
 * \defgroup alloc Test allocation of a single item
 * @{
 */

void test_arena_allocator_api_alloc_with_null(void) {
    const size_t size = sizeof(int);
    void *item = arena_allocator_api_alloc(NULL, size);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_alloc_with_zero_size(void) {
    const size_t size = 0U;
    void *item = arena_allocator_api_alloc(&harena, size);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_alloc_with_null_and_zero_size(void) {
    const size_t size = 0U;
    void *item = arena_allocator_api_alloc(NULL, size);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_alloc_address(void) {
    int *item = (int *)arena_allocator_api_alloc(&harena, sizeof(item));
    TEST_ASSERT_EQUAL_PTR(item, harena.items[0U].value);
}

void test_arena_allocator_api_alloc_value(void) {
    int *item = (int *)arena_allocator_api_alloc(&harena, sizeof(item));
    *item = 10;
    TEST_ASSERT_EQUAL_INT(*item, *(int *)harena.items[0U].value);
}

/*! @} */

/*!
 * \defgroup calloc Test allocation of multiple items
 * @{
 */

void test_arena_allocator_api_calloc_with_null(void) {
    const size_t size = sizeof(int);
    const size_t count = 3U;
    void *item = arena_allocator_api_calloc(NULL, size, count);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_calloc_with_zero_size(void) {
    const size_t size = 0U;
    const size_t count = 3U;
    void *item = arena_allocator_api_calloc(&harena, size, count);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_calloc_with_zero_count(void) {
    const size_t size = sizeof(int);
    const size_t count = 0U;
    void *item = arena_allocator_api_calloc(&harena, size, count);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_calloc_with_null_and_zero_size(void) {
    const size_t size = 0U;
    const size_t count = 3U;
    void *item = arena_allocator_api_calloc(NULL, size, count);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_calloc_with_null_and_zero_count(void) {
    const size_t size = sizeof(int);
    const size_t count = 0U;
    void *item = arena_allocator_api_calloc(NULL, size, count);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_calloc_with_zero_size_and_zero_count(void) {
    const size_t size = 0U;
    const size_t count = 0U;
    void *item = arena_allocator_api_calloc(&harena, size, count);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_calloc_with_null_and_zero_size_and_zero_count(void) {
    const size_t size = 0U;
    const size_t count = 0U;
    void *item = arena_allocator_api_calloc(NULL, size, count);
    TEST_ASSERT_NULL(item);
}

void test_arena_allocator_api_calloc_address(void) {
    const size_t count = 3U;
    int *item = (int *)arena_allocator_api_calloc(&harena, sizeof(item), count);
    TEST_ASSERT_EQUAL_PTR(item, harena.items[0].value);
}

void test_arena_allocator_api_calloc_values(void) {
    const size_t count = 3U;
    int *item = (int *)arena_allocator_api_calloc(&harena, sizeof(item), count);
    for (size_t i = 0U; i < count; ++i)
        item[i] = (int)i;
    TEST_ASSERT_EQUAL_INT_ARRAY(item, (int *)harena.items[0].value, count);
}

/*! @} */

/*!
 * \defgroup calloc Test allocation of multiple items
 * @{
 */

void test_arena_allocator_api_free_with_null(void) {
    arena_allocator_api_free(NULL);
    /*!
     * The assertion is always true since passing NULL as parameter to the free
     * function sohuld not do anything.
     */
    TEST_ASSERT_TRUE(1);
}

void test_arena_allocator_api_free_with_null_arena(void) {
    arena_allocator_api_free(&harena);
    /*!
     * The assertion is always true since deallocating a non existing arena
     * should do nothing
     */
    TEST_ASSERT_TRUE(1);
}

void test_arena_allocator_api_free_with_null_item(void) {
    int *ptr = (int *)arena_allocator_api_alloc(&harena, sizeof(int));
    free(ptr);
    harena.items[0].value = NULL;
    arena_allocator_api_free(&harena);
    /*!
     * The assertion is always true since deallocating a non existing item
     * should do nothing
     */
    TEST_ASSERT_TRUE(1);
}

void test_arena_allocator_api_free_size(void) {
    (void)arena_allocator_api_alloc(&harena, sizeof(int));
    arena_allocator_api_free(&harena);
    TEST_ASSERT_EQUAL_size_t(0U, harena.size);
}

void test_arena_allocator_api_free_capacity(void) {
    (void)arena_allocator_api_alloc(&harena, sizeof(int));
    arena_allocator_api_free(&harena);
    TEST_ASSERT_EQUAL_size_t(0U, harena.capacity);
}
void test_arena_allocator_api_free_address(void) {
    (void)arena_allocator_api_alloc(&harena, sizeof(int));
    arena_allocator_api_free(&harena);
    TEST_ASSERT_EQUAL_PTR(NULL, harena.items);
}

/*! @} */

int main(void) {
    UNITY_BEGIN();

    /*!
     * \addtogroup item_push Run test for item puh without allocation
     * @{
     */

    RUN_TEST(test_arena_allocator_api_item_push_address);
    RUN_TEST(test_arena_allocator_api_item_push_arena_size);

    /*! @} /

    /*! 
     * \addtogroup item_push_with_alloc Run test for item push with allocation
     * @{
     */
    RUN_TEST(test_arena_allocator_api_item_push_with_alloc_when_zero_capacity_arena_address);
    RUN_TEST(test_arena_allocator_api_item_push_with_alloc_when_zero_capacity_capacity);
    RUN_TEST(test_arena_allocator_api_item_push_with_alloc_when_full_arena_address);
    RUN_TEST(test_arena_allocator_api_item_push_with_alloc_when_full_capacity);
    RUN_TEST(test_arena_allocator_api_item_push_with_alloc_without_realloc_address);
    RUN_TEST(test_arena_allocator_api_item_push_with_alloc_without_realloc_arena_size);

    /*! @} /

    /*! 
     * \addtogroup init Run test for initialization
     * @{
     */

    RUN_TEST(test_arena_allocator_api_init_with_null);
    RUN_TEST(test_arena_allocator_api_init_values);

    /*! @} */

    /*! 
     * \addtogroup alloc Run test for allocation of a single item
     * @{
     */

    RUN_TEST(test_arena_allocator_api_alloc_with_null);
    RUN_TEST(test_arena_allocator_api_alloc_with_zero_size);
    RUN_TEST(test_arena_allocator_api_alloc_with_null_and_zero_size);
    RUN_TEST(test_arena_allocator_api_alloc_address);
    RUN_TEST(test_arena_allocator_api_alloc_value);

    /*! @} */

    /*! 
     * \addtogroup calloc Run test for allocation of multiple items
     * @{
     */

    RUN_TEST(test_arena_allocator_api_calloc_with_null);
    RUN_TEST(test_arena_allocator_api_calloc_with_zero_size);
    RUN_TEST(test_arena_allocator_api_calloc_with_zero_count);
    RUN_TEST(test_arena_allocator_api_calloc_with_null_and_zero_size);
    RUN_TEST(test_arena_allocator_api_calloc_with_null_and_zero_count);
    RUN_TEST(test_arena_allocator_api_calloc_with_zero_size_and_zero_count);
    RUN_TEST(test_arena_allocator_api_calloc_with_null_and_zero_size_and_zero_count);
    RUN_TEST(test_arena_allocator_api_calloc_address);
    RUN_TEST(test_arena_allocator_api_calloc_values);

    /*! @} */

    /*! 
     * \addtogroup free Run test for deallocation
     * @{
     */

    RUN_TEST(test_arena_allocator_api_free_with_null);
    RUN_TEST(test_arena_allocator_api_free_with_null_arena);
    RUN_TEST(test_arena_allocator_api_free_with_null_item);
    RUN_TEST(test_arena_allocator_api_free_size);
    RUN_TEST(test_arena_allocator_api_free_capacity);
    RUN_TEST(test_arena_allocator_api_free_address);

    /*! @} */

    return UNITY_END();
}
