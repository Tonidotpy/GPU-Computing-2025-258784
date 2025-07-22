# SpMV CSR GPU Adaptive Implementation

This implementation uses a **load balancing** mechanism by adaptively choosing
the best method to calculate the matrix-vector product based on the amount of
non-zero values for each rows.

Three different methods are used:
1. CSR-Stream: used for batches of **multiple rows**, process rows efficiently
    using either scalar or logarithmic reduction
2. CSR-VectorL: used for **single dense rows**, utilizing mutliple warps and
    atomic accumulation operations
3. CSR-Vector: applied to single rows that are not dense enough to justify
    warp-level parallelism

## Build From Source

Follow the instruction written inside the parent folder README.

## Usage

Follow the instruction written inside the parent folder README.

Inside the `config.h` file some behavior of the implementation can be changed,
such as:
- The logger log level to only show a specific category of log messages;
- The large matrix threshold used to show a warning message if the number of non-zeros of the matrix is too large;
- A flag used to add colors to the logger output
- The number of warp-up cycles to be run;
- The number of profiled iterations to be run;
- A flag used to dump the input and output vectors into a matrix market file;
- The type used for matrices and vectors values;
- The type used for integer data such as indices or sizes;
- The type used for storing the profiled times.
- The number of thread per workgroup
- The number of non-zero per workgroup as well as a multiplier
- The minimum number of rows to use the CSR-Stream method
