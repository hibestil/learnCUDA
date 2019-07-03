# learnCUDA
This repository contains notes about CUDA programming.

## Terminology
### Heterogeneous Computing
  - Host The CPU and its memory (host memory)
  - Device The GPU and its memory (device memory)
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

## Programming 
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
mykernel<<<1,1>>>();
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
/*
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
In this example we use pointers for the variables. `add()` runs on the device, so *a*, *b* and *c* must point to device memory. We need to allocate memory on the GPU. 

