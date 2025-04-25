#ifndef CONFIG_H
#define CONFIG_H

#include "logger.h"

/*! Default logger log level */
#ifndef DEFAULT_LOG_LEVEL
#define DEFAULT_LOG_LEVEL (LOGGER_LEVEL_DEBUG)
#endif // DEFAULT_LOG_LEVEL

/*!
 * Threshold used to warn about large number of non zeros of the matrix which
 * operations might take some times
 */
#ifndef LARGE_MATRIX_NZ_THRESHOLD
#define LARGE_MATRIX_NZ_THRESHOLD (500000)
#endif // LARGE_MATRIX_NZ_THRESHOLD

/*! Enable or disable logger colors */
#ifndef LOGGER_COLORS_ENABLE
#define LOGGER_COLORS_ENABLE (true)
#endif // LOGGER_COLORS_ENABLE

/*! Total number of iteration to run without profiling */
#ifndef TSKIP
#define TSKIP (4)
#endif // TSKIP

/*! Number of iteration to run (does not take into account TSKIP) */
#ifndef TITER
#define TITER (10)
#endif // TITER

/*! Dump the output array values to file */
// #define DUMP_OUTPUT

/*! \brief matrices and vectors data type */
typedef float dtype_t;

#endif // CONFIG_H
