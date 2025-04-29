#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <stdbool.h>

#include "config.h"

/*!
 * \brief CSR matrix structure definition
 *
 * \var symmetric True if the matrix is symmetric, false otherwise (used to save space)
 * \var row_count Total number of rows
 * \var col_count Total number of columns
 * \var nz Total number of non-zeros
 * \var rows Pointer to the row index array
 * \var cols Pointer to the col index array
 * \var data Pointer to the data array
 */
typedef struct _CsrMatrix_t {
    bool symmetric;
    dsize_t row_count, col_count, nz;

    dsize_t *rows;
    dsize_t *cols;
    dtype_t *data;
} CsrMatrix_t;

/*!
 * \brief Initialize the CSR matrix structure
 *
 * \param mat A pointer to the CSR matrix structure
 */
void csr_matrix_init(CsrMatrix_t *mat);

/*!
 * \brief Check if the matrix is symmetric or not
 *
 * \param mat A pointer to matrix structure
 *
 * \return bool True if the matrix is symmetric, false otherwise
 */
bool csr_is_symmetric(CsrMatrix_t *mat);

/*!
 * \brief Generate matrix row prefix sum from the sorted row indices
 * 
 * \param mat A pointer to the matrix structure
 * \param sorted_rows The array of sorted row indices
 */
void csr_pack(CsrMatrix_t *mat, dsize_t *sorted_rows);

#endif // CSR_MATRIX_H
