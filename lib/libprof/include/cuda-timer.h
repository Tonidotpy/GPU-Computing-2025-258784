#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

/*!
 * \brief Cuda profiling timer handler structure
 *
 * \var start Cuda event of the timer start
 * \var stop Cuda event of the timer stop
 */
typedef struct _CudaTimerHandler_t {
    cudaEvent_t start;
    cudaEvent_t stop;
} CudaTimerHandler_t;

/*!
 * \brief Initialize the timer handler
 *
 * \param htimer A pointer to the timer handler
 */
void cuda_timer_init(CudaTimerHandler_t *htimer);

/*!
 * \brief Start the timer
 *
 * \param htimer A pointer to the timer handler
 */
void cuda_timer_start(CudaTimerHandler_t *htimer);

/*!
 * \brief Stop the timer
 *
 * \param htimer A pointer to the timer handler
 */
void cuda_timer_stop(CudaTimerHandler_t *htimer);

/*!
 * \brief Get the elapsed time of the timer
 *
 * \param htimer A pointer to the timer handler
 *
 * \return double The elapsed time in seconds
 */
double cuda_timer_elapsed(CudaTimerHandler_t *htimer);

/*!
 * \brief Deinitialize the timer handler
 *
 * \param htimer A pointer to the timer handler
 */
void cuda_timer_deinit(CudaTimerHandler_t *htimer);

#endif // CUDA_TIMER_H
