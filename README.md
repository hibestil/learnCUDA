<p align="center">
  <img width="350" height="197" src="https://s3.amazonaws.com/cms.ipressroom.com/219/files/20149/544a8211f6091d588d00014e_NVIDIA_CUDA-web/NVIDIA_CUDA-web_mid.jpg">
</p>

# Learn CUDA
__This repository contains notes and examples__ to get started Parallel Computing with CUDA. 

## Introduction
CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.

In GPU-accelerated applications, the sequential part of the workload runs on the CPU – which is optimized for single-threaded performance – while the compute intensive portion of the application runs on thousands of GPU cores in parallel. When using CUDA, developers program in popular languages such as C, C++, Fortran, Python and MATLAB and express parallelism through extensions in the form of a few basic keywords.

CUDA accelerates applications across a wide range of domains from image processing, to deep learning, numerical analytics and computational science. (ref:https://developer.nvidia.com/cuda-zone) 
## Table of Contents
1. [What we will learn?](#what-will-we-learn)  
1. [Useful Links](#UsefulLinks) 
1. [Background](#Background)
    1. [Terminology](#Terminology)
    1. [GPU Synchronization](#GPUSynchronization)
    1. [Managing the Device](#Managing-the-Device)
        1. [Coordinating Host & Device](#CoordinatingHostDevice)
        1. [Device Management](#DeviceManagement)
        1. [Reporting Errors](#ReportingErrors)
    1. [Memory Management](#MemoryManagement)
        1. [CUDA API for Handling Device Memory](#CUDAAPIforHandlingDeviceMemory)
        1. [Notes about GPU Limitations](#NotesaboutGPULimitations)
        1. [Debugging and Profiling](#DebuggingandProfiling)
1. [Optimization Techniques for CUDA](#OptimizationTechniquesforCUDA)
1. [Examples](#Examples)
    1. [Hello CUDA World](#HelloCUDAWorld)
    1. ["Hello CUDA World!" with Device Code](#HelloCUDAWorldwithDeviceCode)
    1. [Addition on the Device](#AdditionontheDevice)
    1. [Vector Addition on Device (Parallel)](#VectorAdditiononDeviceParallel)
    1. [Vector Addition Using Threads](#VectorAdditionUsingThreads)
    1. [Combining Threads and Blocks](#CombiningThreadsandBlocks)
    1. [Handling Arbitrary Vector Sizes](#HandlingArbitraryVectorSizes)
    1. [Cooperating Threads](#CooperatingThreads)
    1. [Loops & Nested loops](#LoopsNestedloops)
1. [References](#references)
  
  
<a name="what-will-we-learn"></a>
## What we will learn? 
- [x] Write and launch CUDA C/C++ kernels
  - [x] `__global__`, `<<<>>>`, `blockIdx`, `threadIdx`, `blockDim`
- [x] Manage GPU memory
  - [x] `cudaMalloc()`, `cudaMemcpy()`, `cudaFree()`
- [x] Manage communication and synchronization
  - [x] `__shared__`, `__syncthreads()`
  - [x] `cudaMemcpy()` vs `cudaMemcpyAsync()`, `cudaDeviceSynchronize()`
- [x] CUDA Optimization Techniques
- [ ] CUDA Persistent Threads

<a name="UsefulLinks"></a>
## Useful Links
- Introduction to Parallel Computing : https://computing.llnl.gov/tutorials/parallel_comp/
- BOOK : Designing and Building Parallel Programs : https://www.mcs.anl.gov/~itf/dbpp/text/book.html
- BOOK : Introduction to Parallel Computing. https://www-users.cs.umn.edu/~karypis/parbook/
- Performance Metrics in CUDA C/C++ : https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
- CUDA Zone – tools, training and webinars : https://developer.nvidia.com/cuda-zone
- Udacity - Intro to Parallel Programming. : https://www.youtube.com/watch?v=GiGE3QjwknQ&list=PLGvfHSgImk4aweyWlhBXNF6XISY3um82_&index=36
- CUDA Samples : https://github.com/NVIDIA/cuda-samples
- An Easy Introduction to CUDA C and C++ : https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/
- Avoiding Pitfalls when Using NVIDIA GPUs : http://drops.dagstuhl.de/opus/volltexte/2018/8984/
- GPU Activity Monitor – NVIDIA Jetson TX Dev Kit : https://github.com/jetsonhacks/gpuGraphTX
<a name="Background"></a>
# Background
<a name="Terminology"></a>
## Terminology
- Heterogeneous Computing
  - Host The CPU and its memory (host memory)
  - Device The GPU and its memory (device memory)
- The __compute capability__ of a device describes its architecture, e.g.
  - Number of registers
  - Sizes of memories
  - Features & capabilities
- Programming in Parallel
  - GPU computing is about massive parallelism. We will use __'blocks'__ and __'threads'__ to implement parallelism.
- Blocks
  - ...
- Threads
  - A block can be split into parallel threads. Unlike parallel blocks, threads have mechanisms to efficiently:
  - Communicate
  - Synchronize
- What is a thread block?
    - One thread block consists of set of threads. Those threads may be in 1D, 2D or 3D. When we consider a thread block, threadIdx and blockDim standard variables in CUDA can be considered very important.
    - ```cpp
        threadIdx   // = Used to access the index of a thread inside a thread block 
        threadIdx.x // = Index of a thread inside a block in X direction
        threadIdx.y // = Index of a thread inside a block in Y direction
        threadIdx.z // = Index of a thread inside a block in Z direction
        ```
    - ```cpp
        blockDim    // = Number of threads in the block for a specific direction
        blockDim.x  // = Number of threads in the block for X direction
        blockDim.y  // = Number of threads in the block for Y direction
        blockDim.z  // = Number of threads in the block for Z direction
        ```
-  What is a thread grid?
    - Thread grid is a set of thread blocks. Blocks also can be in 1D, 2D or 3D (Imagine replacing threads by thread blocks in the previous clarification for thread blocks). When it comes to thread grid, following variables are important.
    - ```cpp
        blockIdx = Used to access an index of a thread block inside a thread grid
        blockIdx.x = Index of a tread block in X direction
        blockIdx.y = Index of a tread block in Y direction
        blockIdx.z = Index of a tread block in Z direction
        ```
    - ```cpp
        gridDim = Number of thread blocks in a specific direction.
        gridDim.x = Number of thread blocks in X direction
        gridDim.y = Number of thread blocks in Y direction
        gridDim.z = Number of thread blocks in Z direction
        ```
- Launching parallel threads:
  - Launch __N__ blocks with __M__ threads per block with `kernel<<<N,M>>>(…);`
  - Use `blockIdx.x` to access block index within grid
  - Use `threadIdx.x` to access thread index within block

- Allocate elements to threads:
  - ```cpp
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    ```
- Use `__shared__` to declare a variable/array in shared memory
  - Data is shared between threads in a block
  - Not visible to threads in other blocks
- Use `__syncthreads()` as a barrier
  - Use to prevent data hazards

- Sharing Data Between Threads
  - Within a block, threads share data via __shared memory__.
  - Extremely fast on-chip memory,
  >   - By opposition to device memory, referred to as __global memory__
  >   - Like a user-managed cache
  - Declare using __`__shared__`__, allocated per block.
  - Data is not visible to threads in other blocks.
  - __Implementing With Shared Memory__
  >   - Cache data in shared memory
  >   - Read `(blockDim.x + 2 * radius)` input elements from global memory to shared memory
  >   - Compute `blockDim.x` output elements
  >   - Write `blockDim.x` output elements to global memory
<a name="GPUSynchronization"></a>
## GPU Synchronization
- In CUDA there are multiple ways to achieve GPU synchronization. They fall into two broad categories: explicit synchronization, which is always programmer-requested, and implicit synchronization, which can occur as a side effect of CUDA API functions intended for purposes other than synchronization. 
    - Explicit Synchronization
        - Explicit synchronization is typically used after a program has launched one or more asynchronous CUDA kernels or memory transfer operations and must wait for computations to complete
    - Implicit Synchronization
<a name="Managing-the-Device"></a>
## Managing the Device
1. <a name="CoordinatingHostDevice"></a>
    __Coordinating Host & Device__
    - Kernel launches are asynchronous
      - Control returns to the CPU immediately
    - CPU needs to synchronize before consuming the results
      - `cudaMemcpy()`            : Blocks the CPU until the copy is complete
    Copy begins when all preceding CUDA calls have completed
      - `cudaMemcpyAsync()`       : Asynchronous, does not block the CPU
      - `cudaDeviceSynchronize()` : Blocks the CPU until all preceding CUDA calls have completed
<a name="DeviceManagement"></a>
1. __Device Management__
    - Application can query and select GPUs
      - ```cpp
        cudaGetDeviceCount(int *count)
        cudaSetDevice(int device)
        cudaGetDevice(int *device)
        cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
        ```
     - Multiple host threads can share a device
     - A single host thread can manage multiple devices
        - `cudaSetDevice(i)` to select current device
        - `cudaMemcpy(…)` for peer-to-peer copies
<a name="ReportingErrors"></a>
1. __Reporting Errors__
    - All CUDA API calls return an error code (`cudaError_t`)
      - Error in the API call itself
      - Or error in an earlier asynchronous operation (e.g. kernel)
    - Get the error code for the last error:
      - ```cpp
          cudaError_t cudaGetLastError(void)
        ```
    - Get a string to describe the error:
    - ```cpp
        char *cudaGetErrorString(cudaError_t)
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
      ```
   
<a name="MemoryManagement"></a>
## Memory Management
Host and device memory are separate entities:
  - Device pointers point to GPU memory
      - May be passed to/from host code
      - May not be dereferenced in host code
  - Host pointers point to CPU memory
      - May be passed to/from device code
      - May not be dereferenced in device code
      
<a name="CUDAAPIforHandlingDeviceMemory"></a>      
### CUDA API for Handling Device Memory
We can use `cudaMalloc()`,`cudaFree()`,`cudaMemcpy()`. 
These ara similar to the C equivalents `malloc()`, `free()`, `memcpy()`.

<a name="NotesaboutGPULimitations"></a>
### Notes about GPU Limitations
An ideal real-time system would be able to terminate periodic tasks that do not complete by their deadline, as the result of their computation is no longer temporally valid. Current GPUs, including the one in the Jetson, do not provide a mechanism for stopping GPU operations after they are launched without resetting the entire device; GPUs cannot be considered preemptable resources as required by many conventional real-time scheduling algorithms. This creates the undesirable property that if we cannot bound the runtime of a
GPU program, we have no guarantees on when it will terminate.

<a name="DebuggingandProfiling"></a>
## Debugging and Profiling
1. Emulation Mode
    - An executable compiled in device emulation mode (`nvcc-deviceemu`) runs only on the CPU (host) using the CUDA runtime support without requiring GPU nor driver
    - Possibilities in device emulation mode:
        - Use all debugging support available on the CPU side (breakpoints,watchdogs, etc.).
        - Access GPU data from CPU code.
        - Call any CPU function from GPU code (for example, printf) and vice versa.
        - Detect deadlocks due to an improper use of `__syncthreads`.
    - Threads are executed sequentially, so simultaneous accesses to the same memory position from multiple threads produces (potencially) different results.
    - Access to values through GPU pointers on the CPU or CPU pointers on the GPU may produce correct results on emulation mode, but will lead to errors when properly executed on GPU.
    - Results coming from floating-point may differ due to:
        - Different outputs from the compiler.
        - Different instructions set.
        - The use of extended precision operators on intermediate results.

    
<a name="OptimizationTechniquesforCUDA"></a>
# Optimization Techniques for CUDA
1. Zero Copy Memory
    - In CUDA programs, CPU memory and GPU memory are used as if they are physically separate, as in discrete GPUs, by default. When the CPU data is made available to the GPU, it is copied from CPU memory to GPU memory. On systems with integrated GPUs, such as the Jetson, the CPU and the GPU share the same physical memory so when data is copied from CPU memory to GPU memory, it moves from one region of memory to another in the same DRAM banks. 
    - On these systems, it is possible for the CPU and GPU to access the same regions of memory when CUDA programs are implemented with Zero Copy CUDA library functions. Using Zero Copy can reduce the memory requirement of GPU programs by up to half because the CPU and the GPU do not need to maintain separate copies of the data. The CPU has access to the original data. 
    - Instead of making a copy of the original data, the GPU uses a pointer to the CPU’s copy for computation. We measure the impacts of caching and data movement by comparing the runtime of the default program implementation to the runtime of the Zero Copy implementation
    - Mapped, pinned memory (zero-copy) is useful when either:
        - The GPU has no memory on its own and uses RAM anyway
        - You load the data exactly once, but you have a lot of computation to perform on it and you want to hide memory transfer latencies through it.
        - The host side wants to change/add more data, or read the results, while kernel is still running (e.g. communication)
        - The data does not fit into GPU memory
    - Pinned, but not mapped memory is better:
        - When you load or store the data multiple times. For example: you have multiple subsequent kernels, performing the work in steps - there is no need to load the data from host every time.
        - There is not that much computation to perform and loading latencies are not going to be hidden well

1. Optimize Memory Usage
    - Minimize transfers between CPU and GPU. 
        - If you want to increase data bandwidth, use “pinned” memory (responsibly), to take advantage of PCI-express capabilities.
    - Group data transfers between CPU and GPU
        - As latency predominates over data bandwidth.
    - Migrate some functions from CPU to GPU even though they may not exploit too much parallelism
        - If this skips shipping data to GPU and the way back to CPU.
    - Optimize memory access patterns
        - Effective data bandwidth may vary an order of magnitude depending on access pattern if we wisely exploit:
            - Coalesced accesses to global memory (less important in Fermi).
            - Shared memory accesses free of conflicts when using 16 banks.
            - Texture memory accesses (they make use of a cache).
            - Accesses to constant memory which shared the same address.
<a name="Examples"></a>
# Examples

<a name="HelloCUDAWorld"></a>
### "Hello CUDA World"
```cpp
int main(void) {
    printf("Hello World!\n");
    return 0;
}
```
To compile use this commands: (NVIDIA compiler (nvcc) can be used to compile programs with no device code)
  - `$ nvcc hello_world.cu`
  - `$ a.out Hello World!`
  
<a name="HelloCUDAWorldwithDeviceCode"></a>
### "Hello CUDA World!" with Device Code
```cpp
/*
CUDA C/C++ keyword `__global__` indicates a function that:
  - Runs on the device
  - Is called from host code
*/
__global__ void my_cuda_kernel(void) {
}
int main(void) {
    my_cuda_kernel<<<1,1>>>();
    printf("Hello World!\n");
    return 0;
}
```
`nvcc` separates source code into host and device components:
  - Device functions (e.g. my_cuda_kernel()) processed by NVIDIA compiler
  - Host functions (e.g. main()) processed by standard host compiler
      - gcc, cl.exe
      
```cpp 
my_cuda_kernel<<<1,1>>>();
```
Triple angle brackets mark a call from host code to device code.
  - Also called a “kernel launch”
  - We’ll return to the parameters (1,1) in a moment
That’s all that is required to execute a function on the GPU!

<a name="AdditionontheDevice"></a>
### Addition on the Device
```cpp 
/*
__global__ is a CUDA C/C++ keyword meaning 
  1. add() will execute on the device
  2. add() will be called from the host
*/
__global__ void add(int *a, int *b, int *c) {
  *c = *a + *b;
  
int main(void) {
      // host copies of a, b, c
      int a, b, c; 
      // device copies of a, b, c
      int *d_a, *d_b, *d_c; 
      int size = sizeof(int);
      // Allocate space for device copies of a, b, c
      cudaMalloc((void **)&d_a, size);
      cudaMalloc((void **)&d_b, size);
      cudaMalloc((void **)&d_c, size);
      // Setup input values
      a = 2;
      b = 7;
      // Copy inputs to device
      cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
      // Launch add() kernel on GPU
      add<<<1,1>>>(d_a, d_b, d_c);
      // Copy result back to host
      cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
      // Cleanup
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
      return 0;
  }
```
In this example we use pointers for the variables. `add()` runs on the device, so *a*, *b* and *c* must point to device memory. We need to allocate memory on the GPU using `cudaMemcpy()` method. 

<a name="VectorAdditiononDeviceParallel"></a>
### Vector Addition on Device (Parallel)
With `add()` (mentioned in previous example) running in parallel we can do vector addition.
  - Each parallel invocation of `add()` is referred to as a *block*.
  - The *set of blocks* is referred to as a *grid*
  - Each invocation can refer to its *block index* using `blockIdx.x`
    ```cpp
    __global__ void add(int *a, int *b, int *c) {
      c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
      }
    ```
  - By using `blockIdx.x` to index into the array, each block handles *a different element* of the array.
```cpp 
// This kernel handles a differen element of the array.
__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 512
int main(void) {
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b, d_c);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}

```
<a name="VectorAdditionUsingThreads"></a>
### Vector Addition Using Threads
We use `threadIdx.x` instead of `blockIdx.x`.
Let’s change add() to use parallel threads instead of parallel blocks:
  ```cpp 
    __global__ void add(int *a, int *b, int *c) {
        c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
    }  
  ```
  
```cpp 
#define N 512
int main(void) {
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b, d_c);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}

```
<a name="CombiningThreadsandBlocks"></a>
### Combining Threads and Blocks
We’ve seen parallel vector addition using:
- Several blocks with one thread each
- One block with several threads
Let’s adapt vector addition to use both blocks and threads.
```cpp 
// Use the built-in variable blockDim.x for threads per block
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}
#define N (2048*2048)
#define THREADS_PER_BLOCK 512
int main(void) {
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);
    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch add() kernel on GPU
    add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```
<a name="HandlingArbitraryVectorSizes"></a>
### Handling Arbitrary Vector Sizes
Avoid accessing beyond the end of the arrays:
```cpp 
__global__ void add(int *a, int *b, int *c, int n) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
if (index < n)
 c[index] = a[index] + b[index];
}
```
And update the kernel launch:
```cpp 
add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
```
<a name="CooperatingThreads"></a>
### Cooperating Threads
1. 1D Stencil
```cpp
// IMPORTANT NOTE : This example causes race condition. Please do not use and refer to __syncthreads().
__global__ void stencil_1d(int *in, int *out) {
     __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
     int gindex = threadIdx.x + blockIdx.x * blockDim.x;
     int lindex = threadIdx.x + RADIUS;
     // Read input elements into shared memory
     temp[lindex] = in[gindex];
     if (threadIdx.x < RADIUS) {
     temp[lindex - RADIUS] = in[gindex - RADIUS];
     temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
 }
 // Apply the stencil
 int result = 0;
 for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
 result += temp[lindex + offset];
 // Store the result
 out[gindex] = result;
}
```
2. `__syncthreads()`
```
cpp void __syncthreads();
```
- Synchronizes all threads within a block
  - Used to prevent RAW / WAR / WAW hazards
- All threads must reach the barrier
  - In conditional code, the condition must be uniform across the block
So we can implement stencil with sycning threads:
```cpp
__global__ void stencil_1d(int *in, int *out) {
      __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
      int gindex = threadIdx.x + blockIdx.x * blockDim.x;
      int lindex = threadIdx.x + radius;
      // Read input elements into shared memory
      temp[lindex] = in[gindex];
      if (threadIdx.x < RADIUS) {
      temp[lindex – RADIUS] = in[gindex – RADIUS];
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
      }
      // Synchronize (ensure all the data is available)
      __syncthreads();
      // Apply the stencil
      int result = 0;
      for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
      result += temp[lindex + offset];
      // Store the result
      out[gindex] = result;
}
```
<a name="LoopsNestedloops"></a>
### Loops & Nested loops
- One for loop
    - ```cpp
        __global__ void single_loop() {
            int i = threadId.x + blockIdx.x * blockDim.x;
            printf("GPU - i = %d \n",i);
        }
        int main(void) {
            // Single loop in CPU
            for(int i = 0; i <4; i++){
                printf("CPU - i = %d \n",i);
            }
            // Single loop in GPU
            dim3 grid(1,1,1);
            dim3 block(4,1,1);
            single_loop<<<grid,block>>>();
            cudaDeviceSynchronize();
            return 0;
        }
        ```
    - Grid dimension is set to (1, 1, 1). That means only one thread block is created. What is the dimension of the thread block? That has been specified as (4, 1, 1). This means 4 threads are created in x direction. Therefore,
        - blockDim.x = 4, blockDim.y = blockDim.z =1.
        - blockIdx.x = blockIdx.y = blockIdx.z = 0.
        - threadIdx.y = threadIdx.z = 0, threadIdx.x varies 0–3 (inclusive).
    - How I have taken the i value? I have added the thread index for X direction to the multiplication of block index in X direction and block dimension for X direction.
- Two nested for loops
    - ```cpp
        __global__ void two_nested_loops() {
            int i = threadId.x + blockIdx.x * blockDim.x;
            int j = threadId.y + blockIdx.y * blockDim.y;
            printf("GPU - i = %d - j = %d \n",i,j);
        }
        int main(void) {
            // Two nested loops CPU
            for(int i = 0; i <4; i++){
                for(int j = 0; j <4; j++){
                    printf("CPU - i = %d - j = %d \n",i,j);
                }
            }
            // Two nested loops GPU
            dim3 grid(1,1,1);
            dim3 block(4,4,1);
            single_loop<<<grid,block>>>();
            cudaDeviceSynchronize();
            return 0;
        }
        ```
    - ```cpp
        // OR we can do same thing by increasing number of blocks
        __global__ void two_nested_loops() {
            int i = threadId.x + blockIdx.x * blockDim.x;
            int j = threadId.y + blockIdx.y * blockDim.y;
            printf("GPU - i = %d - j = %d \n",i,j);
        }
        int main(void) {
            // Two nested loops in CPU
            for(int i = 0; i <4; i++){
                for(int j = 0; j <4; j++){
                    printf("CPU - i = %d - j = %d \n",i,j);
                }
            }
            // Two nested loops in GPU
            dim3 grid(2,2,1);
            dim3 block(2,2,1);
            single_loop<<<grid,block>>>();
            cudaDeviceSynchronize();
            return 0;
        }
        ```

- Triple nested for loops
    - ```cpp
        __global__ void triple_nested_loops() {
            int i = threadId.x + blockIdx.x * blockDim.x;
            int j = threadId.y + blockIdx.y * blockDim.y;
            int k = threadId.z + blockIdx.z * blockDim.z;
            printf("GPU - i = %d - j = %d - k = %d \n",i,j,k);
        }
        int main(void) {
            // Triple nested loops in CPU
            for(int i = 0; i <4; i++){
                for(int j = 0; j <4; j++){
                    for(int k = 0; k <4; k++){
                        printf("CPU - i = %d - j = %d - k = %d \n",i,j,k);
                    }
                }
            }
            // Triple nested loops in GPU
            dim3 grid(1,1,1);
            dim3 block(4,4,4);
            single_loop<<<grid,block>>>();
            cudaDeviceSynchronize();
            return 0;
        }
        ```
    - ```cpp
        // OR we can use two blocks instead of one block.
        __global__ void triple_nested_loops() {
            int i = threadId.x + blockIdx.x * blockDim.x;
            int j = threadId.y + blockIdx.y * blockDim.y;
            int k = threadId.z + blockIdx.z * blockDim.z;
            printf("GPU - i = %d - j = %d - k = %d \n",i,j,k);
        }
        int main(void) {
            // Triple nested loops in CPU
            for(int i = 0; i <4; i++){
                for(int j = 0; j <4; j++){
                    for(int k = 0; k <4; k++){
                        printf("CPU - i = %d - j = %d - k = %d \n",i,j,k);
                    }
                }
            }
            // Triple nested loops in GPU
            dim3 grid(2,2,2);
            dim3 block(2,2,2);
            single_loop<<<grid,block>>>();
            cudaDeviceSynchronize();
            return 0;
        }
        ```

<a name="references"></a>
# References
https://medium.com/@erangadulshan.14/1d-2d-and-3d-thread-allocation-for-loops-in-cuda-e0f908537a52
https://www.researchgate.net/figure/Translated-CUDA-code-from-triple-nested-loop-mappings_fig5_268521516
