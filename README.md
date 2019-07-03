# learnCUDA
This repository contains notes about CUDA programming.

# Background
## Terminology
- Heterogeneous Computing
  - Host The CPU and its memory (host memory)
  - Device The GPU and its memory (device memory)
### Programming in Parallel
GPU computing is about massive parallelism. We will use 'blocks' and 'threads' to implement parallelism.
### CUDA Threads
A block can be split into parallel threads


## Memory Management
Host and device memory are separate entities:
  - Device pointers point to GPU memory
      - May be passed to/from host code
      - May not be dereferenced in host code
  - Host pointers point to CPU memory
      - May be passed to/from device code
      - May not be dereferenced in device code
      
### CUDA API for Handling Device Memory
We can use `cudaMalloc()`,`cudaFree()`,`cudaMemcpy()`. 
These ara similar to the C equivalents `malloc()`, `free()`, `memcpy()`.


# Examples
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
