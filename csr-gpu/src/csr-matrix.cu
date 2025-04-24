#include "csr-matrix.h"

#include <string.h>

extern "C" {
    #include "arena-allocator-api.h"
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

void csr_pack(CsrMatrix_t *mat, int *sorted_rows) {
    if (mat == NULL || sorted_rows == NULL)
        return;

    // Count number of row indices
    for (int i = 0; i < mat->nz; ++i) {
        int r = sorted_rows[i];
        ++mat->rows[r];
    }

    // Prefix-sum
    int cnt = mat->rows[0];
    mat->rows[0] = 0;
    for (int i = 1; i < mat->row_count; ++i) {
        int r = mat->rows[i];
        mat->rows[i] = cnt;
        cnt += r;
    }
    mat->rows[mat->row_count] = cnt;
}
