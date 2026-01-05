// CUDA interview kernels: matmul, transpose, reduce, conv, multi-stream
// Target GPU: NVIDIA H20 (Hopper class, SM90).
//
// Build:
//   nvcc -O3 -std=c++17 -lineinfo -arch=sm_90 -Xptxas=-v all_in_one.cu -o all_in_one
// If your nvcc supports it and you want H100/Hopper tuned SASS:
//   nvcc -O3 -std=c++17 -lineinfo -arch=sm_90a -Xptxas=-v all_in_one.cu -o all_in_one
//
// Run:
//   ./all_in_one

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#define CHECK_CUDA(call)                                                                         \
  do {                                                                                           \
    cudaError_t _e = (call);                                                                     \
    if (_e != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));     \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

static inline float rand_uniform(uint32_t& state) {
  // xorshift32
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (state & 0x00FFFFFF) / float(0x01000000);  // [0,1)
}

static void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  // C[M,N] = A[M,K] * B[K,N] row-major
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
  }
}

static void cpu_transpose(const float* in, float* out, int rows, int cols) {
  // out[cols,rows] = in[rows,cols]
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[c * rows + r] = in[r * cols + c];
    }
  }
}

static float cpu_reduce_sum(const float* x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += x[i];
  return float(s);
}

static float cpu_reduce2d_sum(const float* x, int rows, int cols) {
  double s = 0.0;
  for (int i = 0; i < rows * cols; ++i) s += x[i];
  return float(s);
}

static void cpu_conv2d_valid(const float* x, const float* w, float* y, int H, int W, int KH,
                             int KW) {
  // single-channel, y is VALID conv: OH=H-KH+1, OW=W-KW+1
  int OH = H - KH + 1;
  int OW = W - KW + 1;
  for (int i = 0; i < OH; ++i) {
    for (int j = 0; j < OW; ++j) {
      float acc = 0.f;
      for (int ki = 0; ki < KH; ++ki) {
        for (int kj = 0; kj < KW; ++kj) {
          acc += x[(i + ki) * W + (j + kj)] * w[ki * KW + kj];
        }
      }
      y[i * OW + j] = acc;
    }
  }
}

// --------------------------------- MatMul ---------------------------------

__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

template <int BM, int BN, int BK>
__global__ void matmul_tiled_smem(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
  // Block computes C tile [BM, BN]
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  int row = blockIdx.y * BM + threadIdx.y;  // threadIdx.y in [0,BM)
  int col = blockIdx.x * BN + threadIdx.x;  // threadIdx.x in [0,BN)

  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    // load A tile
    if (row < M && (k0 + threadIdx.x) < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + (k0 + threadIdx.x)];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.f;
    }
    // load B tile
    if ((k0 + threadIdx.y) < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.f;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// -------------------------------- Transpose --------------------------------

__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows,
                                int cols) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols) return;
  out[c * rows + r] = in[r * cols + c];
}

template <int TILE>
__global__ void transpose_tiled_smem(const float* __restrict__ in, float* __restrict__ out,
                                     int rows, int cols) {
  // classic: avoid bank conflicts by padding shared tile [TILE][TILE+1]
  __shared__ float tile[TILE][TILE + 1];

  int x = blockIdx.x * TILE + threadIdx.x;  // input col
  int y = blockIdx.y * TILE + threadIdx.y;  // input row
  if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  __syncthreads();

  int ox = blockIdx.y * TILE + threadIdx.x;  // output col (was row block)
  int oy = blockIdx.x * TILE + threadIdx.y;  // output row
  if (ox < rows && oy < cols) out[oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
}

// ------------------------------- Reduce Sum ---------------------------------

static __device__ __forceinline__ float warp_reduce_sum(float v) {
  // full mask
  unsigned mask = 0xFFFFFFFFu;
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset);
  return v;
}

__global__ void reduce_sum_naive_atomic(const float* __restrict__ x, float* __restrict__ out,
                                        int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) atomicAdd(out, x[idx]);
}

__global__ void reduce_sum_block_then_atomic(const float* __restrict__ x, float* __restrict__ out,
                                             int n) {
  // each block reduces to one value then atomicAdd once (much less contention)
  float sum = 0.f;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    sum += x[idx];
  }

  // reduce within block using warp shuffles + shared for warp sums
  sum = warp_reduce_sum(sum);
  __shared__ float warp_sums[32];  // up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  float block_sum = 0.f;
  if (warp == 0) {
    block_sum = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) atomicAdd(out, block_sum);
  }
}

// Demonstrate bank-conflict handling in shared reduction:
// - Use "sequential addressing" and avoid interleaved stride patterns.
__global__ void reduce_sum_smem_sequential(const float* __restrict__ x, float* __restrict__ out,
                                          int n) {
  extern __shared__ float s[];
  float local = 0.f;
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x * 2 + tid;
  if (idx < n) local += x[idx];
  if (idx + blockDim.x < n) local += x[idx + blockDim.x];
  s[tid] = local;
  __syncthreads();

  // sequential addressing reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  // warp reduce the last 32 using shuffles (avoid final shared conflicts)
  float v = (tid < 32) ? s[tid] : 0.f;
  if (tid < 32) {
    v = warp_reduce_sum(v);
    if (tid == 0) atomicAdd(out, v);
  }
}

// ------------------------------ Reduce2D Sum --------------------------------

__global__ void reduce2d_sum_flatten(const float* __restrict__ x, float* __restrict__ out,
                                     int rows, int cols) {
  int n = rows * cols;
  // reuse 1D "block then atomic" strategy
  float sum = 0.f;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    sum += x[idx];
  }
  sum = warp_reduce_sum(sum);
  __shared__ float warp_sums[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float block_sum = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) atomicAdd(out, block_sum);
  }
}

// ---------------------------------- Conv -----------------------------------

__global__ void conv2d_valid_naive(const float* __restrict__ x, const float* __restrict__ w,
                                   float* __restrict__ y, int H, int W, int KH, int KW) {
  int OH = H - KH + 1;
  int OW = W - KW + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // col in output
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // row in output
  if (i >= OH || j >= OW) return;
  float acc = 0.f;
  for (int ki = 0; ki < KH; ++ki)
    for (int kj = 0; kj < KW; ++kj) acc += x[(i + ki) * W + (j + kj)] * w[ki * KW + kj];
  y[i * OW + j] = acc;
}

template <int TILE>
__global__ void conv2d_valid_tiled_input(const float* __restrict__ x, const float* __restrict__ w,
                                         float* __restrict__ y, int H, int W, int KH, int KW) {
  // Tile output: TILE x TILE; load input patch: (TILE+KH-1)x(TILE+KW-1)
  int OH = H - KH + 1;
  int OW = W - KW + 1;

  int out_x0 = blockIdx.x * TILE;
  int out_y0 = blockIdx.y * TILE;

  int in_tile_w = TILE + KW - 1;
  int in_tile_h = TILE + KH - 1;
  extern __shared__ float shmem[];
  float* in_tile = shmem;
  float* w_tile = in_tile + in_tile_w * in_tile_h;

  // load weights into shared (small)
  for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < KH * KW;
       t += blockDim.x * blockDim.y) {
    w_tile[t] = w[t];
  }

  // load input patch cooperatively
  for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < in_tile_w * in_tile_h;
       t += blockDim.x * blockDim.y) {
    int ty = t / in_tile_w;
    int tx = t - ty * in_tile_w;
    int in_y = out_y0 + ty;
    int in_x = out_x0 + tx;
    float v = 0.f;
    if (in_y < H && in_x < W) v = x[in_y * W + in_x];
    in_tile[ty * in_tile_w + tx] = v;
  }
  __syncthreads();

  int ox = out_x0 + threadIdx.x;
  int oy = out_y0 + threadIdx.y;
  if (threadIdx.x < TILE && threadIdx.y < TILE && ox < OW && oy < OH) {
    float acc = 0.f;
    #pragma unroll
    for (int ki = 0; ki < 16; ++ki) {  // cap unroll; runtime check breaks but fine
      if (ki >= KH) break;
      #pragma unroll
      for (int kj = 0; kj < 16; ++kj) {
        if (kj >= KW) break;
        acc += in_tile[(threadIdx.y + ki) * in_tile_w + (threadIdx.x + kj)] *
               w_tile[ki * KW + kj];
      }
    }
    y[oy * OW + ox] = acc;
  }
}

// ------------------------------- Multi-stream --------------------------------

__global__ void saxpy(float* __restrict__ y, const float* __restrict__ x, float a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = a * x[idx] + y[idx];
}

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

static void print_device() {
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s\n", prop.name);
  printf("CC: %d.%d, SMs: %d, Mem: %.1f GiB\n", prop.major, prop.minor, prop.multiProcessorCount,
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

static bool allclose(const std::vector<float>& a, const std::vector<float>& b, float rtol,
                     float atol) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    float da = std::fabs(a[i] - b[i]);
    float tol = atol + rtol * std::fabs(b[i]);
    if (da > tol) return false;
  }
  return true;
}

static void test_matmul() {
  printf("\n[1] 矩阵乘 (MatMul)\n");
  const int M = 512, N = 512, K = 512;
  size_t bytesA = size_t(M) * K * sizeof(float);
  size_t bytesB = size_t(K) * N * sizeof(float);
  size_t bytesC = size_t(M) * N * sizeof(float);

  std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
  uint32_t st = 1;
  for (auto& v : hA) v = rand_uniform(st) - 0.5f;
  for (auto& v : hB) v = rand_uniform(st) - 0.5f;

  cpu_matmul(hA.data(), hB.data(), hRef.data(), M, N, K);

  float *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, bytesA));
  CHECK_CUDA(cudaMalloc(&dB, bytesB));
  CHECK_CUDA(cudaMalloc(&dC, bytesC));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

  dim3 block1(16, 16);
  dim3 grid1((N + block1.x - 1) / block1.x, (M + block1.y - 1) / block1.y);

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));

  // naive
  CHECK_CUDA(cudaMemset(dC, 0, bytesC));
  CHECK_CUDA(cudaEventRecord(s));
  matmul_naive<<<grid1, block1>>>(dA, dB, dC, M, N, K);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
  printf("  - naive: %.3f ms, correct=%s\n", elapsed_ms(s, e),
         allclose(hC, hRef, 1e-3f, 1e-3f) ? "YES" : "NO");

  // tiled shared memory
  // Using 16x16 block, BK=16 -> shared tiles [16,16] and [16,16]
  CHECK_CUDA(cudaMemset(dC, 0, bytesC));
  CHECK_CUDA(cudaEventRecord(s));
  matmul_tiled_smem<16, 16, 16><<<grid1, block1>>>(dA, dB, dC, M, N, K);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
  printf("  - tiled+smem: %.3f ms, correct=%s\n", elapsed_ms(s, e),
         allclose(hC, hRef, 1e-3f, 1e-3f) ? "YES" : "NO");

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
}

static void test_transpose() {
  printf("\n[2] 矩阵转置 (Transpose)\n");
  const int rows = 1024, cols = 1024;
  size_t bytes = size_t(rows) * cols * sizeof(float);

  std::vector<float> hIn(rows * cols), hOut(cols * rows), hRef(cols * rows);
  uint32_t st = 7;
  for (auto& v : hIn) v = rand_uniform(st) - 0.5f;
  cpu_transpose(hIn.data(), hRef.data(), rows, cols);

  float *dIn, *dOut;
  CHECK_CUDA(cudaMalloc(&dIn, bytes));
  CHECK_CUDA(cudaMalloc(&dOut, bytes));
  CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));

  dim3 block(32, 8);
  dim3 grid((cols + 31) / 32, (rows + 31) / 32);

  // naive
  CHECK_CUDA(cudaEventRecord(s));
  transpose_naive<<<grid, block>>>(dIn, dOut, rows, cols);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));
  printf("  - naive: %.3f ms, correct=%s\n", elapsed_ms(s, e),
         allclose(hOut, hRef, 1e-3f, 1e-3f) ? "YES" : "NO");

  // tiled shared (bank-conflict free via +1 padding)
  CHECK_CUDA(cudaEventRecord(s));
  transpose_tiled_smem<32><<<grid, dim3(32, 8)>>>(dIn, dOut, rows, cols);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));
  printf("  - tiled+smem(pad): %.3f ms, correct=%s\n", elapsed_ms(s, e),
         allclose(hOut, hRef, 1e-3f, 1e-3f) ? "YES" : "NO");

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(dIn));
  CHECK_CUDA(cudaFree(dOut));
}

static void test_reduce1d() {
  printf("\n[3] 一维 reduce-sum (重点：shared memory bank conflict)\n");
  const int n = 1 << 24;  // ~16M
  size_t bytes = size_t(n) * sizeof(float);

  std::vector<float> hX(n);
  uint32_t st = 123;
  for (auto& v : hX) v = rand_uniform(st) - 0.5f;
  float ref = cpu_reduce_sum(hX.data(), n);

  float* dX;
  float* dOut;
  CHECK_CUDA(cudaMalloc(&dX, bytes));
  CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));

  // naive atomicAdd per element (very slow)
  CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));
  int block = 256;
  int grid = (n + block - 1) / block;
  grid = std::min(grid, 65535);
  CHECK_CUDA(cudaEventRecord(s));
  reduce_sum_naive_atomic<<<grid, block>>>(dX, dOut, n);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float out = 0.f;
  CHECK_CUDA(cudaMemcpy(&out, dOut, sizeof(float), cudaMemcpyDeviceToHost));
  printf("  - naive atomic/elem: %.3f ms, err=%.3e\n", elapsed_ms(s, e), std::fabs(out - ref));

  // block reduce then single atomic
  CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));
  CHECK_CUDA(cudaEventRecord(s));
  reduce_sum_block_then_atomic<<<grid, block>>>(dX, dOut, n);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(&out, dOut, sizeof(float), cudaMemcpyDeviceToHost));
  printf("  - block reduce + warp shfl: %.3f ms, err=%.3e\n", elapsed_ms(s, e),
         std::fabs(out - ref));

  // shared reduction (sequential addressing) + warp shfl tail
  CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));
  int grid2 = (n + (block * 2 - 1)) / (block * 2);
  grid2 = std::min(grid2, 65535);
  size_t shmem = size_t(block) * sizeof(float);
  CHECK_CUDA(cudaEventRecord(s));
  reduce_sum_smem_sequential<<<grid2, block, shmem>>>(dX, dOut, n);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(&out, dOut, sizeof(float), cudaMemcpyDeviceToHost));
  printf("  - smem sequential + shfl: %.3f ms, err=%.3e\n", elapsed_ms(s, e),
         std::fabs(out - ref));

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dOut));
}

static void test_reduce2d() {
  printf("\n[4] 二维 reduce-sum\n");
  const int rows = 4096, cols = 4096;
  const int n = rows * cols;
  size_t bytes = size_t(n) * sizeof(float);
  std::vector<float> hX(n);
  uint32_t st = 999;
  for (auto& v : hX) v = rand_uniform(st) - 0.5f;
  float ref = cpu_reduce2d_sum(hX.data(), rows, cols);

  float* dX;
  float* dOut;
  CHECK_CUDA(cudaMalloc(&dX, bytes));
  CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));

  int block = 256;
  int grid = 65535;  // oversubscribe, each block grid-stride
  CHECK_CUDA(cudaEventRecord(s));
  reduce2d_sum_flatten<<<grid, block>>>(dX, dOut, rows, cols);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float out = 0.f;
  CHECK_CUDA(cudaMemcpy(&out, dOut, sizeof(float), cudaMemcpyDeviceToHost));
  printf("  - flatten grid-stride reduce: %.3f ms, err=%.3e\n", elapsed_ms(s, e),
         std::fabs(out - ref));

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dOut));
}

static void test_conv2d() {
  printf("\n[5] 卷积 (2D 单通道 VALID)\n");
  const int H = 1024, W = 1024;
  const int KH = 3, KW = 3;
  const int OH = H - KH + 1;
  const int OW = W - KW + 1;
  size_t bytesX = size_t(H) * W * sizeof(float);
  size_t bytesW = size_t(KH) * KW * sizeof(float);
  size_t bytesY = size_t(OH) * OW * sizeof(float);

  std::vector<float> hX(H * W), hW(KH * KW), hY(OH * OW), hRef(OH * OW);
  uint32_t st = 42;
  for (auto& v : hX) v = rand_uniform(st) - 0.5f;
  for (auto& v : hW) v = rand_uniform(st) - 0.5f;
  cpu_conv2d_valid(hX.data(), hW.data(), hRef.data(), H, W, KH, KW);

  float *dX, *dW, *dY;
  CHECK_CUDA(cudaMalloc(&dX, bytesX));
  CHECK_CUDA(cudaMalloc(&dW, bytesW));
  CHECK_CUDA(cudaMalloc(&dY, bytesY));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), bytesX, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dW, hW.data(), bytesW, cudaMemcpyHostToDevice));

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));

  dim3 block(16, 16);
  dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y);

  // naive
  CHECK_CUDA(cudaEventRecord(s));
  conv2d_valid_naive<<<grid, block>>>(dX, dW, dY, H, W, KH, KW);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(hY.data(), dY, bytesY, cudaMemcpyDeviceToHost));
  printf("  - naive: %.3f ms, correct=%s\n", elapsed_ms(s, e),
         allclose(hY, hRef, 1e-3f, 1e-3f) ? "YES" : "NO");

  // tiled input into shared
  constexpr int TILE = 16;
  dim3 grid2((OW + TILE - 1) / TILE, (OH + TILE - 1) / TILE);
  dim3 block2(16, 16);
  int in_tile_w = TILE + KW - 1;
  int in_tile_h = TILE + KH - 1;
  size_t shmem = (in_tile_w * in_tile_h + KH * KW) * sizeof(float);
  CHECK_CUDA(cudaEventRecord(s));
  conv2d_valid_tiled_input<TILE><<<grid2, block2, shmem>>>(dX, dW, dY, H, W, KH, KW);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  CHECK_CUDA(cudaMemcpy(hY.data(), dY, bytesY, cudaMemcpyDeviceToHost));
  printf("  - tiled input+smem: %.3f ms, correct=%s\n", elapsed_ms(s, e),
         allclose(hY, hRef, 1e-3f, 1e-3f) ? "YES" : "NO");

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dW));
  CHECK_CUDA(cudaFree(dY));
}

static void test_multistream() {
  printf("\n[6] 单 stream 改多 stream（H2D/D2H 拷贝与计算重叠示例）\n");

  const int n = 1 << 24;  // 16M
  size_t bytes = size_t(n) * sizeof(float);
  const int num_streams = 4;
  const int chunk = (n + num_streams - 1) / num_streams;

  // pinned host buffers for async copies
  float *hX, *hY;
  CHECK_CUDA(cudaMallocHost(&hX, bytes));
  CHECK_CUDA(cudaMallocHost(&hY, bytes));

  uint32_t st = 2026;
  for (int i = 0; i < n; ++i) {
    hX[i] = rand_uniform(st) - 0.5f;
    hY[i] = rand_uniform(st) - 0.5f;
  }

  float *dX, *dY;
  CHECK_CUDA(cudaMalloc(&dX, bytes));
  CHECK_CUDA(cudaMalloc(&dY, bytes));

  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; ++i) CHECK_CUDA(cudaStreamCreate(&streams[i]));

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));

  // baseline: single stream sync copies + kernel
  CHECK_CUDA(cudaEventRecord(s));
  CHECK_CUDA(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, hY, bytes, cudaMemcpyHostToDevice));
  int block = 256;
  int grid = (n + block - 1) / block;
  saxpy<<<grid, block>>>(dY, dX, 2.0f, n);
  CHECK_CUDA(cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms_single = elapsed_ms(s, e);
  printf("  - single stream total: %.3f ms\n", ms_single);

  // multi-stream: chunked async H2D, kernel, D2H per stream
  CHECK_CUDA(cudaEventRecord(s));
  for (int si = 0; si < num_streams; ++si) {
    int offset = si * chunk;
    int len = std::min(chunk, n - offset);
    if (len <= 0) continue;
    CHECK_CUDA(cudaMemcpyAsync(dX + offset, hX + offset, size_t(len) * sizeof(float),
                               cudaMemcpyHostToDevice, streams[si]));
    CHECK_CUDA(cudaMemcpyAsync(dY + offset, hY + offset, size_t(len) * sizeof(float),
                               cudaMemcpyHostToDevice, streams[si]));
    int gridc = (len + block - 1) / block;
    saxpy<<<gridc, block, 0, streams[si]>>>(dY + offset, dX + offset, 2.0f, len);
    CHECK_CUDA(cudaMemcpyAsync(hY + offset, dY + offset, size_t(len) * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[si]));
  }
  for (int si = 0; si < num_streams; ++si) CHECK_CUDA(cudaStreamSynchronize(streams[si]));
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms_multi = elapsed_ms(s, e);
  printf("  - %d streams chunked total: %.3f ms\n", num_streams, ms_multi);

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  for (auto& stm : streams) CHECK_CUDA(cudaStreamDestroy(stm));
  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dY));
  CHECK_CUDA(cudaFreeHost(hX));
  CHECK_CUDA(cudaFreeHost(hY));
}

int main() {
  print_device();
  // warmup
  CHECK_CUDA(cudaFree(0));

  test_matmul();
  test_transpose();
  test_reduce1d();
  test_reduce2d();
  test_conv2d();
  test_multistream();

  CHECK_CUDA(cudaDeviceSynchronize());
  printf("\nDONE.\n");
  return 0;
}

