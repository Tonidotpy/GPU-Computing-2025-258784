# SpMV CSR GPU Shared Memory Implementation

This implementation exploits **shared memory** to improve throughput during the
Sparse Matrix-Vector multiplication using the standard CSR matrix format.

The matrix data is subdivided into fixed block of 32 threads each, for every
thread a single multiplication is calculated and stored inside the shared
memory.

Once all the threads of the block have calculated the product a parallel
**prefix sum** is constructed for each block directly into the shared memory.

Then the last element of each block (i.e. the items at index $32 \cdot i + 31$)
and of each row (i.e. given the i-th row with $j \in [n, n + m[$ the item with
index $n + m - 1$) is copied into global memory to calculate the final result.

Finally a second kernel is used to calculate the result for each row $i$ as
follows:
```python
def y(x, i, r):
    j_i = math.floor(r[i] / 32) * 32
    j_ii = math.floor(r[i + 1] / 32) * 32

    sum = 0
    for j in range(j_i, j_ii, 32):
        sum += x[j + 31]

    if j_ii != r[i + 1]:
        sum += x[j_ii - 1]

    if j_i != r[i]:
        sum -= x[j_i - 1]

    return sum
```
where $y$ is the output, $x$ is the input prefix sums and $r$ is the list of
non-zero row indices.

This algorithm takes advantage of the prefix sum characteristics since the sum
of any range $[n, n+m]$ can be calculated by subtracting the $(n - 1)$-th item to
the $(n + m)$-th item.

The summation is required since the prefix sum is partially calculated for each
block.

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
