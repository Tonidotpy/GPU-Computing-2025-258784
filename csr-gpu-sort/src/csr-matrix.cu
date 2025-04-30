#include "csr-matrix.h"

extern "C" {
#include <string.h>
#include <math.h>

#include "logger.h"
#include "common.h"
#include "arena-allocator-api.h"
}

__global__ void bitonic_sort_step(CsrMatrix_t *mat, int j, int k) {
    dsize_t i = threadIdx.x + blockDim.x * blockIdx.x;
    dsize_t l = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if (l > i) {
        /* Sort ascending */
        if ((i & k) == 0 && mat->rows[i] > mat->rows[l]) {
            SWAP(dsize_t, mat->rows[i], mat->rows[l]);
            SWAP(dsize_t, mat->cols[i], mat->cols[l]);
            SWAP(dtype_t, mat->data[i], mat->data[l]);
        }
        if ((i & k) != 0 && mat->rows[i] < mat->rows[l]) {
            /* Sort descending */
            SWAP(dsize_t, mat->rows[i], mat->rows[l]);
            SWAP(dsize_t, mat->cols[i], mat->cols[l]);
            SWAP(dtype_t, mat->data[i], mat->data[l]);
        }
    }
}

void csr_matrix_init(CsrMatrix_t *mat) {
    if (mat == NULL)
        return;
    memset(mat, 0, sizeof(*mat));
}

bool csr_is_symmetric(CsrMatrix_t *mat) {
    if (mat == NULL)
        return false;
    return mat->symmetric;
}

extern LoggerHandler_t hlogger;
void csr_sort(CsrMatrix_t *mat) {
    const dsize_t len = pow(2, (dsize_t)ceil(log2(mat->nz)));
    logger_debug(&hlogger, "len = %lu\n", len);

    // Prepare and copy device data
    CsrMatrix_t *d_mat;
    dsize_t *d_rows, *d_cols;
    dtype_t *d_data;
    cudaMallocManaged(&d_mat, sizeof(*d_mat));
    cudaMalloc(&d_rows, len * sizeof(*d_rows));
    cudaMalloc(&d_cols, len * sizeof(*d_cols));
    cudaMalloc(&d_data, len * sizeof(*d_data));

    cudaMemcpy(d_mat, mat, sizeof(*d_mat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, mat->rows, mat->nz * sizeof(*d_rows), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, mat->cols, mat->nz * sizeof(*d_cols), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, mat->data, mat->nz * sizeof(*d_data), cudaMemcpyHostToDevice);

    // Prepare and copy padding
    const dsize_t delta = len - mat->nz;
    logger_debug(&hlogger, "delta = %lu\n", delta);
    dsize_t *idx_pad = (dsize_t *)malloc(delta * sizeof(*idx_pad));
    dtype_t *data_pad = (dtype_t *)malloc(delta * sizeof(*data_pad));
    for (dsize_t i = 0U; i < delta; ++i) {
        idx_pad[i] = MAX(mat->row_count, mat->col_count) + 1;
        data_pad[i] = INFINITY;
    }
    cudaMemcpy(d_rows + mat->nz, idx_pad, delta * sizeof(*d_rows), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols + mat->nz, idx_pad, delta * sizeof(*d_cols), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data + mat->nz, data_pad, delta * sizeof(*d_data), cudaMemcpyHostToDevice);
    free(idx_pad);
    free(data_pad);

    d_mat->rows = d_rows;
    d_mat->cols = d_cols;
    d_mat->data = d_data;

    const dsize_t blocks = len < 512U ? 1 : MIN(MAX_BLOCK_COUNT, len / 512U);
    const dsize_t thread_per_block = MIN(MAX_THREAD_COUNT, len / blocks);
    logger_debug(&hlogger, "blocks = %lu, threads per block = %lu\n", blocks, thread_per_block);

    /* Major step */
    for (dsize_t k = 2U; k <= len; k <<= 1U) {
        /* Minor step */
        for (dsize_t j = k >> 1U; j > 0; j >>= 1U) {
            bitonic_sort_step<<<blocks, thread_per_block>>>(d_mat, j, k);

            cudaDeviceSynchronize();
        }
    }

    // Copy data back to host
    cudaMemcpy(mat->rows, d_rows, mat->nz * sizeof(*d_rows), cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->cols, d_cols, mat->nz * sizeof(*d_cols), cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->data, d_data, mat->nz * sizeof(*d_data), cudaMemcpyDeviceToHost);

    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_data);
}

void csr_pack(CsrMatrix_t *mat, dsize_t *sorted_rows) {
    if (mat == NULL || sorted_rows == NULL)
        return;

    // Count number of row indices
    for (dsize_t i = 0; i < mat->nz; ++i) {
        dsize_t r = sorted_rows[i];
        ++mat->rows[r];
    }

    // Prefix-sum
    dsize_t cnt = mat->rows[0];
    mat->rows[0] = 0;
    for (dsize_t i = 1; i < mat->row_count; ++i) {
        dsize_t r = mat->rows[i];
        mat->rows[i] = cnt;
        cnt += r;
    }
    mat->rows[mat->row_count] = cnt;
}
