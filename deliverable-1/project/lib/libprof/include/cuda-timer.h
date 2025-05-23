#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

/*!
 * \brief Time conversion constants
 */
#define US_TO_S (1.0 / 1e6)
#define MS_TO_S (1.0 / 1e3)
#define S_TO_US (1e6)
#define S_TO_MS (1e3)

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
 * \brief Synchronize cuda threads and events
 *
 * \param htimr A pointer to the timer handler
 */
void cuda_timer_synchronize(CudaTimerHandler_t *htimer);

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
