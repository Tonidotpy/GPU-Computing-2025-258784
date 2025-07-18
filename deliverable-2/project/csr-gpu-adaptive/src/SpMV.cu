#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <math.h>

extern "C" {
#include "config.h"
#include "common.h"
#include "logger.h"
#include "arena-allocator-api.h"
#include "mmio.h"
#include "prof-timer.h"
#include "profiling.h"
}

#include "csr-matrix.h"
#include "cuda-timer.h"

LoggerHandler_t hlogger;
ArenaAllocatorHandler_t harena;
CsrMatrix_t mat;
ProfilingData prof_data;

void output_dump(char filename[128], dtype_t *y, dsize_t count);

/*!
 * \brief Exit point of the program
 *
 * \details This function is used to deinitialize everything before the program
 *      ends
 *
 * \param exit_code The exit code value
 * \param fmt String that will be formatted and printed
 * \params ... Additional parameters needed to format the string
 */
void panic(int exit_code, const char *fmt, ...) {
    arena_allocator_api_free(&harena);

    if (fmt != NULL) {
        va_list args;
        va_start(args, fmt);
        // Print message to stdout if the exit code is 0, stderr otherwise
        FILE *out = exit_code == EXIT_SUCCESS ? stdout : stderr;
        vfprintf(out, fmt, args);
        va_end(args);
    }
    exit(exit_code);
}

/*!
 * \brief Interrupt signal handler
 *
 * \param signo Signal number
 * \param info Additional info about the signal
 * \param ucontext Signal context information
 */
void int_handler(int signo, siginfo_t *info, void *ucontext) {
    UNUSED(signo);
    UNUSED(info);
    UNUSED(ucontext);
    panic(EXIT_FAILURE, NULL);
}

/*!
 * \brief Print usage message and exit
 *
 * \param argc Number of given arguments
 * \param argv List of arguments
 */
void print_usage_and_exit(int argc, char *argv[]) {
    UNUSED(argc);
    panic(EXIT_SUCCESS, "usage: %s [file.mtx]\n", argv[0]);
}

void setup(void) {
    profiling_init(&prof_data);

    ProfTimerHandler_t htimer;
    prof_timer_init(&htimer);
    prof_timer_start(&htimer);

    logger_init(&hlogger, DEFAULT_LOG_LEVEL, LOGGER_COLORS_ENABLE);
    logger_info(&hlogger, "initializing...\n", "");

    arena_allocator_api_init(&harena);
    csr_matrix_init(&mat);

    prof_timer_stop(&htimer);
    prof_data.tsetup = prof_timer_elapsed(&htimer);
}

/*!
 * \brief Parse a Matrix Market file to get the matrix data
 *
 * \param path The path of the file to parse
 */
void parse_matrix_from_file(char *path) {
    ProfTimerHandler_t htimer;
    prof_timer_init(&htimer);

    ProfTimerHandler_t htim_parse;
    prof_timer_init(&htim_parse);

    prof_timer_start(&htimer);

    logger_info(&hlogger, "parsing file %s\n", path);
    printf("Running SpMV on matrix: %s\n", path);

    // Open file
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        panic(EXIT_FAILURE, strerror(errno));
    }

    // Parse initial banner
    MM_typecode matcode;
    const char *err_msg = "error while processing Matrix Market banner from file\n";
    if (mm_read_banner(fp, &matcode) != 0) {
        fclose(fp);
        panic(EXIT_FAILURE, err_msg);
    }

    // Check supported matrix types
    if (!mm_is_real(matcode) && !mm_is_pattern(matcode) && !mm_is_integer(matcode)) {
        logger_error(&hlogger, "matrix data type not supported\n", "");
        fclose(fp);
        panic(EXIT_FAILURE, err_msg);
    }
    if (!mm_is_general(matcode) && !mm_is_symmetric(matcode)) {
        logger_error(&hlogger, "matrix storage scheme not supported\n", "");
        fclose(fp);
        panic(EXIT_FAILURE, err_msg);
    }

    mat.symmetric = mm_is_symmetric(matcode);

    char *typestr = mm_typecode_to_str(matcode);
    logger_debug(&hlogger, "[%s] %s\n", matcode, typestr);
    free(typestr);

    // Get matrix info
    err_msg = "error while processing Matrix Market data from file\n";
    int row_count, col_count, nz;
    if (mm_read_mtx_crd_size(fp, &row_count, &col_count, &nz) != 0) {
        logger_error(&hlogger, "could not process Matrix Market coordinate size\n", "");
        fclose(fp);
        panic(EXIT_FAILURE, err_msg);
    }
    mat.row_count = row_count;
    mat.col_count = col_count;
    mat.nz = nz;

    const char *info_fmt = "\n\n    +---------- MATRIX INFO ----------+\n"
                           "    |                                 |\n"
                           "    |   o Symmetric: %14s   |\n"
                           "    |   o Rows: %19d   |\n"
                           "    |   o Columns: %16d   |\n"
                           "    |   o Non-zeros: %14d   |\n"
                           "    \\_________________________________/\n\n";
    logger_info(&hlogger, info_fmt, mat.symmetric ? "Yes" : "No", mat.row_count, mat.col_count, mat.nz);
    if (mat.nz > LARGE_MATRIX_NZ_THRESHOLD)
        logger_warning(&hlogger, "loading matrix with a large number of non-zeros!!!\n", "");

    // Allocate memory for the matrix data
    prof_timer_start(&htim_parse);

    logger_info(&hlogger, "allocating memory for the matrix...\n", "");
    mat.rows = (dsize_t *)arena_allocator_api_calloc(&harena, sizeof(*mat.rows), mat.nz * 2);
    mat.cols = (dsize_t *)arena_allocator_api_calloc(&harena, sizeof(*mat.cols), mat.nz * 2);
    mat.data = (dtype_t *)arena_allocator_api_calloc(&harena, sizeof(*mat.data), mat.nz * 2);
    if (mat.rows == NULL || mat.cols == NULL || mat.data == NULL) {
        logger_error(&hlogger, "could not allocate enough memory for the matrix data\n", "");
        fclose(fp);
        panic(EXIT_FAILURE, strerror(errno));
    }
    memset(mat.rows, 0, mat.nz * 2 * sizeof(*mat.rows));
    memset(mat.cols, 0, mat.nz * 2 * sizeof(*mat.cols));
    memset(mat.data, 0, mat.nz * 2 * sizeof(*mat.data));

    prof_timer_stop(&htim_parse);
    prof_data.tparse.allocation = prof_timer_elapsed(&htim_parse);

    // Parse matrix data from file line by line
    logger_info(&hlogger, "parsing Matrix Market file data...\n", "");
    for (dsize_t i = 0; i < mat.nz; ++i) {
        int r, c;
        double real = 1, imm = 1;

        prof_timer_start(&htim_parse);

        if (mm_read_mtx_crd_entry(fp, &r, &c, &real, &imm, matcode) != 0) {
            logger_error(&hlogger, "could not parse Matrix Market data\n", "");
            fclose(fp);
            panic(EXIT_FAILURE, err_msg);
        }

        prof_timer_stop(&htim_parse);
        prof_data.tparse.io += prof_timer_elapsed(&htim_parse);

        if (i % MAX(1, (mat.nz / 10)) == 0) {
            logger_debug(&hlogger, "progress %.0f%%\n", (float)i / mat.nz * 100.f);
        }

        /*! Rows and columns indices starts from 1 */
        mat.rows[i] = r - 1;
        mat.cols[i] = c - 1;
        mat.data[i] = real;

        /*
         * Update floating point operation count
         * The operation are a multiplication and an addition for each non-zero
         * Symmetric matrices has to be taken in account
         */
        prof_data.flop += 2U;
        if (csr_is_symmetric(&mat) && r != c) {
            prof_data.flop += 2U;
            ++i;
            ++mat.nz;

            /*! Rows and columns indices starts from 1 */
            mat.rows[i] = c - 1;
            mat.cols[i] = r - 1;
            mat.data[i] = real;
        }
    }

    fclose(fp);

    prof_timer_stop(&htimer);
    prof_data.tparse.total = prof_timer_elapsed(&htimer);

    logger_debug(&hlogger, "parsing done!!!\n", "");
}

void construct_csr_matrix(void) {
    ProfTimerHandler_t htimer;
    prof_timer_init(&htimer);

    ProfTimerHandler_t htim_csr;
    prof_timer_init(&htim_csr);

    prof_timer_start(&htimer);

    logger_info(&hlogger, "constructing CSR matrix...\n", "");

    prof_timer_start(&htim_csr);

    // Sort rows
    csr_sort(&mat);

    prof_timer_stop(&htim_csr);
    prof_data.tcsr.sort = prof_timer_elapsed(&htim_csr);

    logger_info(&hlogger, "generating rows prefix sum...\n", "");

    prof_timer_start(&htim_csr);

    dsize_t *rows = mat.rows;
    mat.rows = (dsize_t *)arena_allocator_api_calloc(&harena, sizeof(*mat.rows), mat.row_count + 1);
    memset(mat.rows, 0, sizeof(*mat.rows) * (mat.row_count + 1));

    // Pack matrix rows
    csr_pack(&mat, rows);

    prof_timer_stop(&htim_csr);
    prof_data.tcsr.pack = prof_timer_elapsed(&htim_csr);

    prof_timer_stop(&htimer);
    prof_data.tcsr.total = prof_timer_elapsed(&htimer);
}

dtype_t *generate_input_vector(dsize_t count) {
    ProfTimerHandler_t htimer;
    prof_timer_init(&htimer);
    prof_timer_start(&htimer);

    logger_info(&hlogger, "generating input vector...\n", "");
    dtype_t *x = (dtype_t *)arena_allocator_api_calloc(&harena, sizeof(*x), count);
    for (dsize_t i = 0; i < count; ++i) {
        x[i] = (rand() % RAND_MAX) / 1e6;
    }

    prof_timer_stop(&htimer);
    prof_data.tgen = prof_timer_elapsed(&htimer);
    return x;
}

void generate_row_blocks(
    const CsrMatrix_t *mat,
    dsize_t **row_blocks_out,
    dsize_t **row_meta_out,
    dsize_t *num_blocks_out) {
    const dsize_t row_count = mat->row_count;
    const dsize_t est_max_blocks = row_coun * 4U;
    dsize_t *row_blocks = arena_allocator_api_calloc(&harena, sizeof(*row_blocks), est_max_blocks + 1U);
    dsize_t *row_meta = arena_allocator_api_calloc(&harena, sizeof(*row_blocks), est_max_blocks);
    dsize_t current_row = 0U;
    dsize_t block_idx = 0U;

    while (current_row < row_count) {
        const dsize_t nnz = mat->rows[current_row + 1U] - mat->rows[current_row];

        if (nnz > NNZ_PER_WG * NNZ_MULTIPLIER) {
            const dsize_t wg_count = (nnz + NNZ_PER_WG * NNZ_MULTIPLIER - 1U) / (NNZ_PER_WG * NNZ_MULTIPLIER);
            for (dsize_t w = 0U; w < wg_count; ++w) {
                row_blocks[block_idx] = current_row;
                row_meta[block_idx] = (current_row << 16U) | w;
                ++block_id;
            }
            ++current_row;
        } else {
            const dsize_t first = current_row;
            dsize_t cnt = 0U;

            while (current_row < row_count && cnt < NUM_ROWS_STREAM) {
                ++current_row;
                ++cnt;
            }
            row_blocks[block_idx] = first;
            row_meta[block_idx] = 0U;
            ++block_id;
        }
    }
    row_blocks[block_idx] = row_count; // Final stop row

    *row_blocks_out = row_blocks;
    *row_meta_out = row_meta;
    *num_blocks_out = block_idx;
}

__device__ void scalar_reduction(
    const CsrMatrix_t *mat,
    const dtype_t *x,
    dtype_t *y,
    const dsize_t r_first,
    const dsize_t r_last) {
    dsize_t local_row = r_first + threadIdx.x;
    while (local_row < r_last) {
        dtype_t sum = 0;
        const dsize_t r_f = mat->rows[local_row];
        const dsize_t r_l = mat->rows[local_row + 1U];
        for (dsize_t i = r_f; i < r_l; ++i) {
            sum += mat->data[i] * x[mat->cols[i]];
        }
        y[local_row] = sum;
        local_row += blockDim.x;
    }
}

__device__ void logarithmic_reduction(
    const CsrMatrix_t *mat,
    const dtype_t *x,
    dtype_t *y,
    const dsize_t r_first,
    const dsize_t r_last,
    dtype_t *cache) {

    const dsize_t local_id = threadIdx.x;
    const dsize_t wg_size = blockDim.x;
    const dsize_t num_rows = r_last - r_first;
    const dsize_t threads_per_row = 1U << (31U - __clz(MAX(1U, wg_size / num_rows))); // Round to power of two

    const dsize_t r_id = local_id / threads_per_row;
    const dsize_t thread_in_row = local_id % threads_per_row;
    if (r_id >= num_rows)
        return;

    const dsize_t r = r_first + r_id;
    const dsize_t r_f = mat->rows[r];
    const dsize_t r_l = mat->rows[r + 1U];
    const dsize_t nnz = r_l - r_f;
    const dsize_t nnz_per_thread = (nnz + threads_per_row - 1U) / threads_per_row;

    dtype_t local_sum = 0;
    for (dsize_t i = 0U; i < nnz_per_thread; ++i) {
        dsize_t idx = r_f + thread_in_row * nnz_per_thread + i;
        if (idx < r_l) {
            local_sum += mat->data[idx] * x[mat->cols[idx]];
        }
    }
    cache[local_id] = local_sum;
    __syncthreads();

    for (dsize_t off = threads_per_row / 2U; off > 0U; off >>= 1U) {
        if (thread_in_row < off) {
            cache[local_id] += cache[local_id + off];
        }
        __syncthreads();
    }

    if (thread_in_row == 0) {
        y[r] = cache[local_id];
    }
}

__device__ void csr_stream(
    const CsrMatrix_t *mat,
    const dtype_t *x,
    dtype_t *y,
    const dsize_t r_first,
    const dsize_t r_last,
    dtype_t *cache) {
    const dsize_t num_rows = r_last - r_first;
    const dsize_t threads_per_row = blockDim.x / num_rows;

    if (threads_per_row >= NUM_ROWS_STREAM) {
        logarithmic_reduction(mat, x, y, r_first, r_last, cache);
    } else {
        scalar_reduction(mat, x, y, r_first, r_last);
    }
}

__device__ void csr_vector(
    const CsrMatrix_t *mat,
    const dtype_t *x,
    dtype_t *y,
    const dsize_t row) {
    const dsize_t local_id = threadIdx.x;
    const dsize_t r_first = mat->rows[row];
    const dsize_t r_last = mat->rows[row + 1U];
    const dsize_t nnz = r_last - r_first;

    __shared__ dtype_t cache[WG_SIZE];
    dtype_t local_sum = 0;

    for (dsize_t i = local_id; i < nnz; i += blockDim.x) {
        const dsize_t idx = r_first + i;
        local_sum += mat->data[idx] * x[mat->cols[idx]];
    }

    cache[local_id] = local_sum;
    __syncthreads();

    for (dsize_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (local_id < stride)
            cache[local_id] += cache[local_id + stride];
        __syncthreads();
    }
    if (local_id == 0U)
        y[row] = cache[0U];
}

__device__ void csr_vector_l(
    const CsrMatrix_t *mat,
    const dtype_t *x,
    dtype_t *y,
    const dsize_t row,
    const dsize_t wg_id,
    const dsize_t wg_count) {
    const dsize_t local_id = threadIdx.x;
    const dsize_t r_first = mat->rows[row];
    const dsize_t r_last = mat->rows[row + 1U];
    const dsize_t nnz = r_last - r_first;
    const dsize_t nnz_per_wg = NNZ_PER_WG * NNZ_MULTIPLIER;

    const dsize_t offset = r_first + wg_id * nnz_per_wg;
    const dsize_t last = MIN(offset + nnz_per_wg, r_last);

    dtype_t sum = 0;
    for (dsize_t i = offset + local_id; i < last; i += blockDim.x) {
        sum += mat->data[i] * x[mat->cols[i]];
    }

    __shared__ dtype_t cache[WG_SIZE];
    cache[local_id] = sum;
    __syncthreads();

    for (dsize_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (local_id < stride)
            cache[local_id] += cache[local_id + stride];
        __syncthreads();
    }
    if (local_id == 0U)
        atomicAdd(&y[row], cache[0U]);
}

__global__ void adaptive_spmv(
    const CsrMatrix_t *mat,
    const dtype_t *x,
    dtype_t *y,
    const dsize_t *row_blocks,
    const dsize_t *row_meta) {
    __shared__ dtype_t cache[WG_SIZE];
    const dsize_t wg_id = blockIdx.x;
    const dsize_t r_first = row_blocks[wg_id];
    const dsize_t r_last = row_blocks[wg_id + 1U];
    const dsize_t num_rows = r_last - r_first;

    if (num_rows >= NUM_ROWS_STREAM) {
        csr_stream(mat, x, y, r_first, r_last, cache);
    } else if (num_rows == 1U) {
        const dsize_t r = r_first;
        const dsize_t r_f = mat->rows[r];
        const dsize_t r_l = mat->rows[r + 1U];
        const dsize_t nnz = r_l - r_f;

        if (nnz > NNZ_PER_WG * NNZ_MULTIPLIER) {
            const dsize_t wg_count = (nnz + NNZ_PER_WG * NNZ_MULTIPLIER - 1U) / (NNZ_PER_WG * NNZ_MULTIPLIER);
            const dsize_t local_wg_id = row_meta[wg_id] & 0xFFFF;
            csr_vector_l(mat, x, y, row, local_wg_id, wg_count);
        } else {
            csr_vector(mat, x, y, row);
        }
    }
}

dtype_t *dispatch(
    const CsrMatrix_t *mat,
    const dtype_t *x,
    const dsize_t *row_blocks,
    const dsize_t *row_meta,
    const dsize_t num_blocks) {
    ProfTimerHandler_t htimer;
    prof_timer_init(&htimer);

    CudaTimerHandler_t htim_spmv;
    cuda_timer_init(&htim_spmv);

    prof_timer_start(&htimer);

    logger_info(&hlogger, "calculating sparse matrix vector product...\n", "");

    ProfTimerHandler_t htim_alloc;
    prof_timer_init(&htim_alloc);
    prof_timer_start(&htim_alloc);

    CsrMatrix_t *d_mat;
    dsize_t *d_rows, *d_cols;
    dtype_t *d_data, *d_x;
    dsize_t *d_row_blocks, *d_row_meta;
    cudaMallocManaged(&d_mat, sizeof(*d_mat));
    cudaMalloc(&d_rows, (mat->row_count + 1) * sizeof(*d_rows));
    cudaMalloc(&d_cols, mat->nz * sizeof(*d_cols));
    cudaMalloc(&d_data, mat->nz * sizeof(*d_data));
    cudaMalloc(&d_x, mat->col_count * sizeof(*d_x));
    cudaMalloc(&d_row_blocks, (num_blocks + 1U) * sizeof(*d_row_blocks));
    cudaMalloc(&d_row_meta, num_blocks * sizeof(*d_row_meta));

    cudaMemcpy(d_mat, mat, sizeof(*d_mat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, mat->rows, (mat->row_count + 1) * sizeof(*d_rows), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(*d_cols), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, mat->data, mat->nz * sizeof(*d_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, mat->col_count * sizeof(*d_x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_blocks, row_blocks, (num_blocks + 1U) * sizeof(*d_row_blocks), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_meta, row_meta, (num_blocks + 1U) * sizeof(*d_row_meta), cudaMemcpyHostToDevice);

    d_mat->rows = d_rows;
    d_mat->cols = d_cols;
    d_mat->data = d_data;

    dtype_t *y = (dtype_t *)arena_allocator_api_calloc(&harena, sizeof(*y), mat->row_count);

    prof_timer_stop(&htim_alloc);
    prof_data.tspmv.allocation = prof_timer_elapsed(&htim_alloc);

    logger_debug(&hlogger, "rows: host %p = device %p [%s]\n", mat->rows, d_rows, mat->rows == d_rows ? "EQUAL" : "NOT EQUAL");
    logger_debug(&hlogger, "cols: host %p = device %p [%s]\n", mat->cols, d_cols, mat->cols == d_cols ? "EQUAL" : "NOT EQUAL");
    logger_debug(&hlogger, "data: host %p = device %p [%s]\n", mat->data, d_data, mat->data == d_data ? "EQUAL" : "NOT EQUAL");
    logger_debug(&hlogger, "Adaptive SpMV Blocks/Threads: <%lu, %lu>\n", num_blocks, WG_SIZE);
    for (dint_t i = -TSKIP; i < TITER; ++i) {
        memset(y, 0, mat->row_count * sizeof(*y));

        // Calculate product
        cuda_timer_start(&htim_spmv);

        adaptive_spmv<<<num_blocks, WG_SIZE>>>(
            d_mat,
            d_x,
            d_y,
            d_row_blocks,
            d_row_meta);

        cuda_timer_synchronize(&htim_spmv);
        cuda_timer_stop(&htim_spmv);

        if (i >= 0) {
            prof_data.tspmv.t[i] = cuda_timer_elapsed(&htim_spmv);
            logger_debug(&hlogger, "iteration %d: %2.5f s\n", i + 1, prof_data.tspmv.t[i]);
        } else {
            logger_debug(&hlogger, "warm-up %d: %2.5f s\n", TSKIP + i + 1, cuda_timer_elapsed(&htim_spmv));
        }
    }

    prof_timer_stop(&htimer);
    prof_data.tspmv.total = prof_timer_elapsed(&htimer);

    cuda_timer_deinit(&htim_spmv);
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_row_blocks);
    cudaFree(d_row_meta);
    return y;
}

void output_dump(char filename[128], dtype_t *y, dsize_t count) {
    const dsize_t len = 256;
    char path[len];
    memset(path, 0, len * sizeof(*path));
    strncpy(path, filename, 128);

    // Open output file
    const time_t t = time(NULL);
    struct tm *tp = localtime(&t);
    strftime(path + strlen(path), len, "-%F-%T.mtx", tp);

    FILE *fp = fopen(path, "w+");
    if (fp == NULL) {
        logger_error(&hlogger, strerror(errno), "");
        return;
    }

    // Write banner
    MM_typecode matcode;
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);
    if (mm_write_banner(fp, matcode) != 0) {
        logger_error(&hlogger, "failed to write output banner to file\n", "");
        fclose(fp);
        return;
    }

    // Write size
    if (mm_write_mtx_array_size(fp, count, 1) != 0) {
        logger_error(&hlogger, "failed to write output array size to file\n", "");
        fclose(fp);
        return;
    }

    // Write data
    for (dsize_t i = 0; i < count; ++i) {
        if (fprintf(fp, "%f\n", y[i]) < 0) {
            logger_error(&hlogger, "failed to write output data to file\n", "");
            fclose(fp);
            return;
        }
    }

    fclose(fp);
}

int main(int argc, char *argv[]) {
    {
        /* Setup signal handling */
        struct sigaction act = { 0 };
        act.sa_flags = SA_SIGINFO;
        act.sa_sigaction = &int_handler;
        if (sigaction(SIGINT, &act, NULL) == -1) {
            panic(EXIT_FAILURE, NULL);
        }
    }

    ProfTimerHandler_t htimer;
    prof_timer_init(&htimer);
    prof_timer_start(&htimer);

    /*  1. Check arguments                                                   */
    if (argc != 2) {
        print_usage_and_exit(argc, argv);
    }

    /*  2. Initialize everything                                             */
    setup();

    /*  3. Read matrix from file                                             */
    parse_matrix_from_file(argv[1]);

    /*  4. Construct matrix with CSR format                                  */
    construct_csr_matrix();

    /*  5. Generate random vector                                            */
    dtype_t *x = generate_input_vector(mat.col_count);

    /*  6. Generate row blocks and meta arrays                               */
    dsize_t *row_blocks;
    dsize_t *row_meta;
    dsize_t num_blocks;
    generate_row_blocks(&mat, &row_blocks, &row_meta, &num_blocks);

    /*  7. Calculate matrix-vector product                                   */
    dtype_t *y = dispatch(&mat, x, row_blocks, row_meta, num_blocks);

    prof_timer_stop(&htimer);
    prof_data.ttotal = prof_timer_elapsed(&htimer);

    /*  8. Print results                                                     */
    profiling_dump(&prof_data);

#ifdef DUMP_OUTPUT
    const dsize_t len = 128;
    char filename[len];
    memset(filename, 0, len * sizeof(*filename));
    strncpy(filename, "input-dump", len);
    output_dump(filename, x, mat.col_count);
    strncpy(filename, "output-dump", len);
    output_dump(filename, y, mat.row_count);
#else
    UNUSED(y);
#endif // DUMP_OUPUT

    panic(EXIT_SUCCESS, NULL);
    return 0;
}
