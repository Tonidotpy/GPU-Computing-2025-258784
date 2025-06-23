#include "profiling.h"

#include <string.h>

#include "statistic.h"

void profiling_init(ProfilingData *data) {
    if (data == NULL)
        return;
    memset(data, 0, sizeof(*data));
}

void profiling_dump(ProfilingData *data) {
    if (data == NULL)
        return;
    const char *prof_fmt = "\n\n    +---------------- SUMMARY --------------------+\n"
                           "    |                                             |\n"
                           "    |   Total time:        %13.6f s        |\n"
                           "    |     1. Setup:        %13.6f s        |\n"
                           "    |     2. Parsing:      %13.6f s        |\n"
                           "    |       a. Allocation: %13.6f s        |\n"
                           "    |       b. I/O:        %13.6f s        |\n"
                           "    |     3. CSR:          %13.6f s        |\n"
                           "    |       a. Sorting:    %13.6f s        |\n"
                           "    |       b. Packing:    %13.6f s        |\n"
                           "    |     4. Input:        %13.6f s        |\n"
                           "    |     5. SpMV:         %13.6f s        |\n"
                           "    |       a. Allocation: %13.6f s        |\n"
                           "    |       b. Mean:       %13g s        |\n"
                           "    |       c. Variance:   %13g          |\n"
                           "    |                                             |\n"
                           "    |   FLOPs:             %13.6f GFLOP/s  |\n"
                           "    \\_____________________________________________/\n\n";

    double mu = gmean(data->tspmv.t, TITER);
    double sigma = var2(data->tspmv.t, TITER, mu);
    double throughput = flops(data->flop, mu) / 1e9;
    printf(
        prof_fmt,
        data->ttotal,
        data->tsetup,
        data->tparse.total,
        data->tparse.allocation,
        data->tparse.io,
        data->tcsr.total,
        data->tcsr.sort,
        data->tcsr.pack,
        data->tgen,
        data->tspmv.total,
        data->tspmv.allocation,
        mu,
        sigma,
        throughput);
}
