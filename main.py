import time

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Define the CUDA kernel for vector addition
kernel_code = """
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Compile the kernel code
mod = SourceModule(kernel_code, options=["-Wno-deprecated-gpu-targets"])

# Get the kernel function from the compiled module
vector_add = mod.get_function("vector_add")

# Define the size of the vectors
n = 1000

# Create input vectors on the host (CPU)
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)

t0 = time.time()

# Allocate memory for the result vector on the host
c = np.zeros_like(a)

# Allocate memory on the device (GPU)
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy data from host to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Define block and grid sizes
block_size = 128
grid_size = (n + block_size - 1) // block_size

# Launch the kernel on the GPU
vector_add(a_gpu, b_gpu, c_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

# Copy the result back to the host
cuda.memcpy_dtoh(c, c_gpu)
t1 = time.time()

# Do the equivalent NumPy computation
d = a + b
t2 = time.time()

# Print the results
print("Vector A:", a)
print("Vector B:", b)
print("Vector C (A + B):", c, "dt:", t1 - t0)
print("Vector D (A + B):", d, "dt:", t2 - t1)
