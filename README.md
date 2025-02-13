This program applies a 3x3 top-hat filter to a randomly generated 2D array. The boundaries are treated as periodic in both directions. The code is written in _C++_ on an _MPI+Kokkos_ framework.

---

# Required packages
*   _cmake 3.10+_ 
*   _Kokkos_ package
*   _CUDA-Aware MPI_ package

# Building and running the code

### Linux
We can follow the standard cmake procedure for building the code and mpirun for code execution.
```sh
mkdir build; cd build
cmake ..
make
mpirun -np 4 ./topHat
```

# Testing the code
The code is tested in Linux machines with A100 and A5000 GPUs that has _Kokkos_ version 4.3.01 and _openmpi_ version 5.0.3. The _openmpi_ package is built with CUDA Aware support.