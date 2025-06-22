#ifndef PROF_TIMER_H
#define PROF_TIMER_H

#include <sys/time.h>

/*!
 * \brief Profiling timer handler structure
 *
 * \var start Timestamp of the start time of the timer
 * \var stop Timestamp of the stop time of the timer
 */
typedef struct _ProfTimerHandler_t {
    struct timeval start;
    struct timeval stop;
} ProfTimerHandler_t;

/*!
 * \brief Initialize the timer handler
 *
 * \param htimer A pointer to the timer handler
 */
void prof_timer_init(ProfTimerHandler_t *htimer);

/*!
 * \brief Start the timer
 *
 * \param htimer A pointer to the timer handler
 */
void prof_timer_start(ProfTimerHandler_t *htimer);

/*!
 * \brief Stop the timer
 *
 * \param htimer A pointer to the timer handler
 */
void prof_timer_stop(ProfTimerHandler_t *htimer);

/*!
 * \brief Get the elapsed time of the timer
 *
 * \param htimer A pointer to the timer handler
 *
 * \return double The elapsed time in seconds
 */
double prof_timer_elapsed(ProfTimerHandler_t *htimer);

#endif // PROF_TIMER_H
