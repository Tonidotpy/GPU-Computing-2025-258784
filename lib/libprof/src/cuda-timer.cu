#include "cuda-timer.h"

#include <string.h>

/*!
 * \brief Time conversion constants
 */
const double us_to_s = 1.0 / 1e6;
const double ms_to_s = 1.0 / 1e3;
const double s_to_us = 1e6;
const double s_to_ms = 1e3;

void cuda_timer_init(CudaTimerHandler_t *htimer) {
    if (htimer == NULL)
        return;
    memset(htimer, 0, sizeof(*htimer));
    cudaEventCreate(&htimer->start);
    cudaEventCreate(&htimer->stop);
}

void cuda_timer_start(CudaTimerHandler_t *htimer) {
    if (htimer != NULL)
        cudaEventRecord(htimer->start);
}

void cuda_timer_stop(CudaTimerHandler_t *htimer) {
    if (htimer != NULL) {
        cudaEventRecord(htimer->stop);
        cudaEventSynchronize(htimer->stop);
    }
}

double cuda_timer_elapsed(CudaTimerHandler_t *htimer) {
    if (htimer == NULL)
        return 0;
    float ms = 0;
    cudaEventElapsedTime(&ms, htimer->start, htimer->stop);
    return ms * ms_to_s;
}

void cuda_timer_deinit(CudaTimerHandler_t *htimer) {
    if (htimer == NULL)
        return;
    cudaEventDestroy(htimer->start);
    cudaEventDestroy(htimer->stop);
}
