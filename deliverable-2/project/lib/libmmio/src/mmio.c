/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>

#include "mmio.h"

#define MM_BUFFER_COUNT (1 << 15)

int mm_pos, mm_len;
char mm_buffer[MM_BUFFER_COUNT];

char mm_read_char(FILE *f) {
    if (mm_pos == mm_len) {
        mm_pos = 0;
        mm_len = (int)fread(mm_buffer, 1, MM_BUFFER_COUNT, f);
        if (mm_len == 0)
            return EOF;
    }
    return mm_buffer[mm_pos++];
}

int mm_read_int(FILE *f, int *x) {
    char c;
    int sign = 1;
    while (!isdigit(c = mm_read_char(f))) {
        if (c == '-')
            sign = -1;
    }
    *x = c - '0';
    while (isdigit(c = mm_read_char(f))) {
        *x = *x * 10 + (c - '0');
    }
    *x *= sign;
    return c == EOF ? -1 : 0;
}

int mm_read_double(FILE *f, double *x) {
    char c;
    double sign = 1;
    // Parse sign
    while (!isdigit(c = mm_read_char(f))) {
        if (c == '-')
            sign = -1;
        else if (c == '.') {
            *x = 0;
            break;
        }
    }
    // Some format omits the unit part if it is 0
    if (isdigit(c)) {
        // Parse digits
        *x = c - '0';
        while (isdigit(c = mm_read_char(f))) {
            *x = *x * 10.0 + (c - '0');
        }
    }
    // Parse decimals
    if (c == '.') {
        double dec = 0.1;
        while (isdigit(c = mm_read_char(f))) {
            *x = *x + (c - '0') * dec;
            dec *= 0.1;
        }
    }
    // Parse exponent
    double e = 1;
    if (c == 'e') {
        bool is_negative = false;
        if ((c = mm_read_char(f)) == '-')
            is_negative = true;
        else
            e = c - '0';

        while (isdigit(c = mm_read_char(f))) {
            e = e * 10.0 + (c - '0');
        }
        if (is_negative)
            e = 1.0 / e;
    }
    *x *= sign * e;
    return c == EOF ? -1 : 0;
}

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_, double **val_, int **I_, int **J_) {
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    double *val;
    int *I, *J;

    if ((f = fopen(fname, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0) {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }

    if (!(mm_is_real(matcode) && mm_is_matrix(matcode) &&
          mm_is_sparse(matcode))) {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return -1;
    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }

    *M_ = M;
    *N_ = N;
    *nz_ = nz;

    /* reseve memory for matrices */

    I = (int *)malloc(nz * sizeof(int));
    J = (int *)malloc(nz * sizeof(int));
    val = (double *)malloc(nz * sizeof(double));

    *val_ = val;
    *I_ = I;
    *J_ = J;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nz; i++) {
        (void)mm_read_int(f, &I[i]);
        (void)mm_read_int(f, &J[i]);
        (void)mm_read_double(f, &val[i]);
    }
    fclose(f);

    return 0;
}

int mm_is_valid(MM_typecode matcode) {
    if (!mm_is_matrix(matcode))
        return 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode))
        return 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode))
        return 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) ||
                                   mm_is_skew(matcode)))
        return 0;
    return 1;
}

int mm_read_banner(FILE *f, MM_typecode *matcode) {
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;

    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p = mtx; *p != '\0'; *p = tolower(*p), p++)
        ; /* convert to lower case */
    for (p = crd; *p != '\0'; *p = tolower(*p), p++)
        ;
    for (p = data_type; *p != '\0'; *p = tolower(*p), p++)
        ;
    for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++)
        ;

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);

    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */

    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else if (strcmp(crd, MM_DENSE_STR) == 0)
        mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;
}

int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz) {
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz) {
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do {
        if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return MM_PREMATURE_EOF;
    } while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;

    else
        do {
            num_items_read = fscanf(f, "%d %d %d", M, N, nz);
            if (num_items_read == EOF)
                return MM_PREMATURE_EOF;
        } while (num_items_read != 3);

    return 0;
}

int mm_read_mtx_array_size(FILE *f, int *M, int *N) {
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;

    /* now continue scanning until you reach the end-of-comments */
    do {
        if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
            return MM_PREMATURE_EOF;
    } while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;

    else /* we have a blank line */
        do {
            num_items_read = fscanf(f, "%d %d", M, N);
            if (num_items_read == EOF)
                return MM_PREMATURE_EOF;
        } while (num_items_read != 2);

    return 0;
}

int mm_write_mtx_array_size(FILE *f, int M, int N) {
    if (fprintf(f, "%d %d\n", M, N) <= 0)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

/*-------------------------------------------------------------------------*/

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[], double val[], MM_typecode matcode) {
    (void)M;
    (void)N;
    bool is_complex = mm_is_complex(matcode);
    bool is_real = mm_is_real(matcode);
    bool is_integer = mm_is_integer(matcode);
    bool is_pattern = mm_is_pattern(matcode);

    if (!is_pattern && !is_integer && !is_real && !is_complex)
        return MM_UNSUPPORTED_TYPE;
    for (int i = 0; i < nz; i++) {
        if (mm_read_int(f, &I[i]) < 0)
            return MM_PREMATURE_EOF;
        if (mm_read_int(f, &J[i]) < 0)
            return MM_PREMATURE_EOF;

        if (is_integer || is_real) {
            if (mm_read_double(f, &val[i]) < 0)
                return MM_PREMATURE_EOF;
        } else if (is_complex) {
            if (mm_read_double(f, &val[2 * i]) < 0)
                return MM_PREMATURE_EOF;
            if (mm_read_double(f, &val[2 * i + 1]) < 0)
                return MM_PREMATURE_EOF;
        }
    }
    return 0;
}

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *imag, MM_typecode matcode) {
    bool is_complex = mm_is_complex(matcode);
    bool is_real = mm_is_real(matcode);
    bool is_integer = mm_is_integer(matcode);
    bool is_pattern = mm_is_pattern(matcode);

    if (!is_pattern && !is_integer && !is_real && !is_complex)
        return MM_UNSUPPORTED_TYPE;
    if (mm_read_int(f, I) < 0)
        return MM_PREMATURE_EOF;
    if (mm_read_int(f, J) < 0)
        return MM_PREMATURE_EOF;

    if (is_integer || is_real || is_complex)
        if (mm_read_double(f, real) < 0)
            return MM_PREMATURE_EOF;

    if (is_complex)
        if (mm_read_double(f, imag) < 0)
            return MM_PREMATURE_EOF;
    return 0;
}

/************************************************************************
    mm_read_mtx_crd()  fills M, N, nz, array of values, and return
                        type code, e.g. 'MCRS'

                        if matrix is complex, values[] is of size 2*nz,
                            (nz pairs of real/imaginary values)
************************************************************************/

int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I, int **J, double **val, MM_typecode *matcode) {
    int ret_code;
    FILE *f;

    if (strcmp(fname, "stdin") == 0)
        f = stdin;
    else if ((f = fopen(fname, "r")) == NULL)
        return MM_COULD_NOT_READ_FILE;

    if ((ret_code = mm_read_banner(f, matcode)) != 0)
        return ret_code;

    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) &&
          mm_is_matrix(*matcode)))
        return MM_UNSUPPORTED_TYPE;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        return ret_code;

    *I = (int *)malloc(*nz * sizeof(int));
    *J = (int *)malloc(*nz * sizeof(int));
    *val = NULL;

    if (mm_is_complex(*matcode)) {
        *val = (double *)malloc(*nz * 2 * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, *matcode);
        if (ret_code != 0)
            return ret_code;
    } else if (mm_is_real(*matcode)) {
        *val = (double *)malloc(*nz * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, *matcode);
        if (ret_code != 0)
            return ret_code;
    }

    else if (mm_is_pattern(*matcode)) {
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, *matcode);
        if (ret_code != 0)
            return ret_code;
    }

    if (f != stdin)
        fclose(f);
    return 0;
}

int mm_write_banner(FILE *f, MM_typecode matcode) {
    char *str = mm_typecode_to_str(matcode);
    int ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
    const int len = strlen(MatrixMarketBanner) + strlen(str) + 2;
    free(str);
    if (ret_code != len)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[], double val[], MM_typecode matcode) {
    FILE *f;
    int i;

    if (strcmp(fname, "stdout") == 0)
        f = stdout;
    else if ((f = fopen(fname, "w")) == NULL)
        return MM_COULD_NOT_WRITE_FILE;

    /* print banner followed by typecode */
    fprintf(f, "%s ", MatrixMarketBanner);
    fprintf(f, "%s\n", mm_typecode_to_str(matcode));

    /* print matrix sizes and nonzeros */
    fprintf(f, "%d %d %d\n", M, N, nz);

    /* print values */
    if (mm_is_pattern(matcode))
        for (i = 0; i < nz; i++)
            fprintf(f, "%d %d\n", I[i], J[i]);
    else if (mm_is_real(matcode))
        for (i = 0; i < nz; i++)
            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
    else if (mm_is_complex(matcode))
        for (i = 0; i < nz; i++)
            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2 * i], val[2 * i + 1]);
    else {
        if (f != stdout)
            fclose(f);
        return MM_UNSUPPORTED_TYPE;
    }

    if (f != stdout)
        fclose(f);

    return 0;
}

/**
*  Create a new copy of a string s.  mm_strdup() is a common routine, but
*  not part of ANSI C, so it is included here.  Used by mm_typecode_to_str().
*
*/
char *mm_strdup(const char *s) {
    int len = strlen(s);
    char *s2 = (char *)malloc((len + 1) * sizeof(char));
    return strcpy(s2, s);
}

char *mm_typecode_to_str(MM_typecode matcode) {
    char buffer[MM_MAX_LINE_LENGTH];
    char *types[4];
    char *mm_strdup(const char *);
    int error = 0;

    /* check for MTX type */
    if (mm_is_matrix(matcode))
        types[0] = MM_MTX_STR;
    else
        error = 1;
    (void)error;

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        return NULL;

    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        return NULL;

    sprintf(buffer, "%s %s %s %s", types[0], types[1], types[2], types[3]);
    return mm_strdup(buffer);
}
