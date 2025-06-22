#include "cuda-timer.h"

#include <string.h>

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

void cuda_timer_synchronize(CudaTimerHandler_t *htimer) {
    if (htimer != NULL) {
        cudaDeviceSynchronize();
        cudaEventSynchronize(htimer->stop);
    }
}

double cuda_timer_elapsed(CudaTimerHandler_t *htimer) {
    if (htimer == NULL)
        return 0;
    float ms = 0;
    cudaEventElapsedTime(&ms, htimer->start, htimer->stop);
    return ms * MS_TO_S;
}

void cuda_timer_deinit(CudaTimerHandler_t *htimer) {
    if (htimer == NULL)
        return;
    cudaEventDestroy(htimer->start);
    cudaEventDestroy(htimer->stop);
}
