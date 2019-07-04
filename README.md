# Learn CUDA
__This repository contains notes and examples about CUDA programming__ to get started with CUDA.
## Introduction
CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.

In GPU-accelerated applications, the sequential part of the workload runs on the CPU – which is optimized for single-threaded performance – while the compute intensive portion of the application runs on thousands of GPU cores in parallel. When using CUDA, developers program in popular languages such as C, C++, Fortran, Python and MATLAB and express parallelism through extensions in the form of a few basic keywords.

CUDA accelerates applications across a wide range of domains from image processing, to deep learning, numerical analytics and computational science. (ref:https://developer.nvidia.com/cuda-zone) 
## Table of Contents
1. [What we will learn?](#what-will-we-learn)  
1. [Useful Links](#UsefulLinks) 
1. [Background](#Background)
    1. [Terminology](#Terminology)
    1. [Managing the Device](#Managing-the-Device)
        1. [Coordinating Host & Device](#CoordinatingHostDevice)
        1. [Device Management](#DeviceManagement)
        1. [Reporting Errors](#ReportingErrors)
    1. [Memory Management](#MemoryManagement)
        1. [CUDA API for Handling Device Memory](#CUDAAPIforHandlingDeviceMemory)
        1. [Notes about GPU Limitations](#NotesaboutGPULimitations)
1. [Examples](#Examples)
    1. [Hello CUDA World](#HelloCUDAWorld)
    1. ["Hello CUDA World!" with Device Code](#HelloCUDAWorldwithDeviceCode)
    1. [Addition on the Device](#AdditionontheDevice)
    1. [Vector Addition on Device (Parallel)](#VectorAdditiononDeviceParallel)
    1. [Vector Addition Using Threads](#VectorAdditionUsingThreads)
    1. [Combining Threads and Blocks](#CombiningThreadsandBlocks)
    1. [Handling Arbitrary Vector Sizes](#HandlingArbitraryVectorSizes)
    1. [Cooperating Threads](#CooperatingThreads)
  
  
<a name="what-will-we-learn"></a>
## What we will learn? 
- [x] Write and launch CUDA C/C++ kernels
  - [x] `__global__`, `<<<>>>`, `blockIdx`, `threadIdx`, `blockDim`
- [x] Manage GPU memory
  - [x] `cudaMalloc()`, `cudaMemcpy()`, `cudaFree()`
- [x] Manage communication and synchronization
  - [x] `__shared__`, `__syncthreads()`
  - [x] `cudaMemcpy()` vs `cudaMemcpyAsync()`, `cudaDeviceSynchronize()`
- [ ] CUDA Optimization Techniques
- [ ] CUDA Persistent Threads

<a name="UsefulLinks"></a>
## Useful Links
- CUDA Zone – tools, training and webinars : https://developer.nvidia.com/cuda-zone
- Udacity - Intro to Parallel Programming. : https://www.youtube.com/watch?v=GiGE3QjwknQ&list=PLGvfHSgImk4aweyWlhBXNF6XISY3um82_&index=36

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
