/*
Compile example:
nvcc -O3 --use_fast_math -Xcompiler -fopenmp -lcublas ./matmul.cu -o matmul

Run example:
OMP_NUM_THREADS=32 ./matmul
*/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

static cublasStatus_t stat;
static cublasHandle_t cublas_handle;

void matmul_cpu(float *A, float *B, float *C, int M, int N, int K) {
#pragma omp parallel for collapse(2)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float val = 0.0f;
      for (int k = 0; k < K; ++k) {
        val += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = val;
    }
  }
}

void matmul_cuda(float *A, float *B, float *C, int M, int N, int K) {
  float alpha = 1.0f;
  float beta = 0.0f;
  stat = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     B, N, A, K, &beta, C, N);
  cublasCheck(stat);
}

int main(int argc, char **argv) {
  srand(0);

  int M = 1024 * 8;
  int N = 768 * 4;
  int K = 768;
  printf("Matrix shape -> M = %d, N = %d, K = %d\n", M, N, K);

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);

  float *A = make_random_float(M * K);
  float *B = make_random_float(K * N);
  float *C = (float *)malloc(M * N * sizeof(float));

  float *A_d;
  float *B_d;
  float *C_d;

  cudaCheck(cudaMalloc(&A_d, M * K * sizeof(float)));
  cudaCheck(cudaMalloc(&B_d, K * N * sizeof(float)));
  cudaCheck(cudaMalloc(&C_d, M * N * sizeof(float)));
  cudaCheck(cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasCheck(cublasCreate(&cublas_handle));
  cublasMath_t cublas_math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

  matmul_cpu(A, B, C, M, N, K);
  matmul_cuda(A_d, B_d, C_d, M, N, K);
  validate_result(C_d, C, "out", M * N, 1e-1f);

  int repeat_times = 100;
  float elapsed_time =
      benchmark_kernel(repeat_times, matmul_cuda, A_d, B_d, C_d, M, N, K);

  float tflops = (float)M * N * K * 2 / elapsed_time * 1e3f / 1e12f;
  printf("time %.4f ms | tflops %.2f\n", elapsed_time, tflops);

  free(A);
  free(B);
  free(C);
  cudaCheck(cudaFree(A_d));
  cudaCheck(cudaFree(B_d));
  cudaCheck(cudaFree(C_d));
  cublasCheck(cublasDestroy(cublas_handle));

  return 0;
}