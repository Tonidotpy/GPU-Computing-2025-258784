#include "statistic.h"

#include <math.h>

double mean(double *x, size_t len) {
    double mu = 0;
    for (size_t i = 0U; i < len; ++i)
        mu += x[i];
    return mu / len;
}

double gmean(double *x, size_t len) {
    double mu = 1;
    for (size_t i = 0U; i < len; ++i)
        mu *= x[i];
    double exp = 1.0 / len;
    return pow(mu, exp);
}

double var(double *x, size_t len) {
    double mu = mean(x, len);
    return var2(x, len, mu);
}

double var2(double *x, size_t len, double mu) {
    double sigma = 0;
    for (size_t i = 0U; i < len; ++i) {
        double aux = x[i] - mu;
        sigma += aux * aux;
    }
    return sigma;
}

double flops(size_t ops_count, double elapsed) {
    return ops_count / elapsed;
}
