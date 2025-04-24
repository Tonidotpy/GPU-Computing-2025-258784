#ifndef STATISTIC_H
#define STATISTIC_H

#include <stddef.h>

/*!
 * \brief Calculate the arithmetic average value of a series
 *
 * \param x The list of values
 * \param len The number of items in the list
 *
 * \return double The average value
 */
double mean(double *x, size_t len);

/*!
 * \brief Calculate the geometric average value of a series
 *
 * \param x The list of values
 * \param len The number of items in the list
 *
 * \return double The average value
 */
double gmean(double *x, size_t len);

/*!
 * \brief Calculate the variance of a series
 *
 * \param x The list of values
 * \param len The number of items in the list
 *
 * \return double The variance
 */
double var(double *x, size_t len);

/*!
 * \brief Calculate the variance of a series
 *
 * \param x The list of values
 * \param len The number of items in the list
 * \param mu The average of the series
 *
 * \return double The variance
 */
double var2(double *x, size_t len, double mu);

/*!
 * \brief Calculate the number of FLOPs
 *
 * \param ops_count Total number of floating point operations
 * \param elapsed The total elapsed time in seconds
 *
 * \return double The number of FLOPs
 */
double flops(size_t ops_count, double elapsed);

#endif // STATISTIC_H
