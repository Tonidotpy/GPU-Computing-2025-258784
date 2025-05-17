#ifndef COMMON_H
#define COMMON_H

/*!
 * GPU limits
 */
#define MAX_THREAD_COUNT (1024U)
#define MAX_BLOCK_COUNT (1024U * 1024U * 64U)

/*!
 * \brief No OPeration, does nothing
 */
#define NOP ((void)0U)

/*!
 * \brief Macro used to prevent compiler warning of unused parameter functions
 *
 * \param _ Unused function argument
 */
#define UNUSED(_) ((void)(_))

/*!
 * \brief Get the minimum value
 *
 * \param a The first value
 * \param b The second value
 *
 * \return The minimum of the two values
 */
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*!
 * \brief Get the maximum value
 *
 * \param a The first value
 * \param b The second value
 *
 * \return The maximum of the two values
 */
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*!
 * \brief Swap two variables of the same type
 *
 * \param type The type of the variables
 * \param a First variable to swap
 * \param b Second variable to swap
 */
#define SWAP(type, a, b) \
    do {                 \
        type c = a;      \
        a = b;           \
        b = c;           \
    } while (0)

#endif // COMMON_H
