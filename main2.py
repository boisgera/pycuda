import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Define the optimized CUDA kernel with grid-stride loops
kernel_code = """
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code, options=["-Wno-deprecated-gpu-targets"])
vector_add = mod.get_function("vector_add")

# Increase problem size to emphasize GPU advantage (3.2GB VRAM used)
n = 200_000_000  # 200 million elements (3 * 4 * 200M = 2.4GB)
print(f"VRAM used: {3 * 4 * n / 2**30:.3f} GiB")

# Create pinned (page-locked) host memory for faster transfers
a_host = cuda.pagelocked_empty(n, dtype=np.float32)
b_host = cuda.pagelocked_empty(n, dtype=np.float32)
c_host = cuda.pagelocked_empty(n, dtype=np.float32)
a_host[:] = np.random.randn(n).astype(np.float32)
b_host[:] = np.random.randn(n).astype(np.float32)

# Optimized kernel configuration
block_size = 256    # Better for modern GPUs than 128
grid_size = 4096    # Balances parallelism and per-thread work

t0 = time.time()

# GPU Memory Operations
a_gpu = cuda.mem_alloc(a_host.nbytes)
b_gpu = cuda.mem_alloc(b_host.nbytes)
c_gpu = cuda.mem_alloc(c_host.nbytes)

t0 = time.time()
cuda.memcpy_htod(a_gpu, a_host)
cuda.memcpy_htod(b_gpu, b_host)

# Warm-up kernel launch to exclude CUDA initialization overhead
#vector_add(a_gpu, b_gpu, c_gpu, np.int32(n), 
#          block=(block_size, 1, 1), grid=(grid_size, 1))

# Time measurement starts after warm-up
vector_add(a_gpu, b_gpu, c_gpu, np.int32(n), 
          block=(block_size, 1, 1), grid=(grid_size, 1))
cuda.Context.synchronize()  # Ensure kernel completes
t1 = time.time()

cuda.memcpy_dtoh(c_host, c_gpu)

# CPU Computation
t2 = time.time()
d = a_host + b_host
t3 = time.time()

# Results
print("NumPy Time:", t3 - t2)
print("GPU Time:", t1 - t0)

speedup = (t3 - t2) / (t1 - t0)
print(f"GPU Computation is {speedup:.1f}x faster than NumPy")
