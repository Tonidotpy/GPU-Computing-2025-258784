# Sparse Matrix-Vector Multiplication

This project contains multiple implementations of the **Sparse Matrix-Vector
multiplication** (SpMV) algorithm leveraging on the **Compressed Sparse Row**
(CSR) matrix representation and also taking advantage of the parallelization
mechanisms offered by NVIDIA Graphic Processing Units (GPUs).

## Project structure

Each folder prefixed with `csr-` contains a different implementation of the
SpMV based on a unique concept, more info can be found inside the
implementation folder itself.

Utilities used alongside this project can be found in the `utils` directory,
those tools are not strictly required for the project to work but they were
used to simplify its development.

Dependencies common to all implementations can be found inside the `lib` folder.

## Build from source

Before building the source make sure to have all the required external
dependecies installed:

- GNU Make: `make`
- NVIDIA CUDA compiler: `nvcc`
- GNU C Compiler: `gcc`

To build the project from source first clone this repo without forgetting to
recursive clone the submodules as well:
```
git clone --recurse-submodules https://github.com/Tonidotpy/GPU-Computing-2025-258784.git
```

A single `Makefile` is provided inside the root folder of the project, which
takes care of the compilation steps of both the different implementations as
well as the library dependencies.

---

To build the project just run:
```
make
```

To avoid building all the project at once a single implementation can be built
given its path, such as:
```
make <path-to-implementation>   # e.g. make csr-cpu
```

The same works for the libraries which can be independently compiled given
their path, such as:
```
make <path-to-library>          # e.g. make lib/libmmio
```

> Each project implementation contains a Makefile which is automatically called
> by the main Makefile upon compilation, even if they should work independently
> it is discouraged the use of them. **Use the methods shown above instead**.

## Usage

A couple of useful parameters can be given to make to change the behavior of
the projects.

The `DEBUG` variable can be set to build the projects with debug information as
follows:
```
make DEBUG=1
```
In this way each compiled programmed can be debugged using the appropriate tools.

The `NLOGGER` variable can be set to build the projects disabling entirely the
logger output as follows:
```
make NLOGGER=1
```
This can be useful to get a clean report from the executable as well as to
completely remove the overhead caused by it (even if it is almost negligible).

Other configuration parameter can be found for each implementation inside the
`config.h` file located in the include directory.
