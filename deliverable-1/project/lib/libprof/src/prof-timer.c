#include "prof-timer.h"

#include <string.h>

/*!
 * \brief Time conversion constants
 */
const double us_to_s = 1.0 / 1e6;
const double ms_to_s = 1.0 / 1e3;
const double s_to_us = 1e6;
const double s_to_ms = 1e3;

void prof_timer_init(ProfTimerHandler_t *htimer) {
    if (htimer == NULL)
        return;
    memset(htimer, 0, sizeof(*htimer));
}

void prof_timer_start(ProfTimerHandler_t *htimer) {
    if (htimer != NULL)
        gettimeofday(&htimer->start, (struct timezone *)0U);
}

void prof_timer_stop(ProfTimerHandler_t *htimer) {
    if (htimer != NULL)
        gettimeofday(&htimer->stop, (struct timezone *)0U);
}

double prof_timer_elapsed(ProfTimerHandler_t *htimer) {
    if (htimer == NULL)
        return 0;
    return (htimer->stop.tv_sec - htimer->start.tv_sec) +
           (htimer->stop.tv_usec - htimer->start.tv_usec) * us_to_s;
}
