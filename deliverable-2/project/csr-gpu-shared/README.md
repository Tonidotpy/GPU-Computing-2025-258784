# SpMV CSR GPU Shared Memory Implementation

<!-- TODO -->

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
