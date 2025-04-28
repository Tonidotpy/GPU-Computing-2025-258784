#ifndef PROFILING_H
#define PROFILING_H

#include "config.h"
#include "common.h"

/*!
 * \brief Structure used to save the profiling data
 */
typedef struct _ProfilingData {
    second_t tsetup;
    struct {
        second_t total;
        second_t allocation;
        second_t io;
        second_t sort;
    } tparse;
    struct {
        second_t total;
        second_t sort;
        second_t pack;
    } tcsr;
    second_t tgen;
    struct {
        second_t total;
        second_t t[TITER];
    } tspmv;

    dsize_t flop;
} ProfilingData;

/*!
 * \brief Initialize profiling data structure
 *
 * \param data A pointer to the profiling data structure
 */
void profiling_init(ProfilingData *data);

/*!
 * \brief Dump the results of the profiling to stdout
 *
 * \param data A pointer to the profiling data
 */
void profiling_dump(ProfilingData *data);

#endif // PROFILING_H
