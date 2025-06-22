# Check Result

This utility is mainly used to check the correctness of the implemented
algorithms.

## Build From Source

To build the project make sure to first install the required dependencies:
- Eigen 12.3.0
- Fast Matrix Market latest (given as submodule)

After installing all the required dependencies build the script using the
provided Makefile by running:
```
make
```

To remove all generated binaries just run:
```
make clean
```

## Usage

To use the script run it by giving as arguments the following three files:
1. `matrix.mtx`: the input sparse matrix in a matrix market file format
2. `in_vector.mtx`: the input vector as a dense vector in the matrix market file format
3. `out_vector.mtx`: the calculated result as a dense vector in the matrix market file format

Once run the script will recalculate the product using the given matrix and
input vector and compare it with the given output to check if they are equal.
If the difference between the calculated and given output is lower than a
certain threshold than the calculation is correct.

> Results may differ based on the used precision (e.g. `float` rather than
> `double`) or truncation of values inside the matrix market files.
