#include "get_brightness_cuda.cuh"
#include <cuda_runtime.h>

// ========== 1. 核函数（Device 执行）==========
__global__ void get_brightness_kernel(
    const float* d_img, int img_rows, int img_cols,
    const float* d_mapx, const float* d_mapy,
    const float* d_kernel, int kernel_size,
    float* d_out, int out_rows, int out_cols
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 输出列
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 输出行

    if (i >= out_rows || j >= out_cols) return;

    int idx = i * out_cols + j;
    float x_f = d_mapx[idx];
    float y_f = d_mapy[idx];

    int x = __float2int_rn(x_f);
    int y = __float2int_rn(y_f);

    int rad = kernel_size / 2;

    // 边界检查
    if (x < rad || y < rad || x >= img_rows - rad || y >= img_cols - rad) {
        d_out[idx] = 0.0f;
        return;
    }

    // 卷积计算
    float sum = 0.0f;
    const float* k = d_kernel;

    for (int dy = 0; dy < kernel_size; ++dy) {
        int img_row = (x - rad + dy);
        int img_base = img_row * img_cols + (y - rad);

        for (int dx = 0; dx < kernel_size; ++dx) {
            sum += d_img[img_base + dx] * k[dy * kernel_size + dx];
        }
    }

    d_out[idx] = sum;
}

// ========== 2. 核函数启动包装（Host 调用）==========
static void launch_get_brightness(
    const float* d_img, int img_h, int img_w,
    const float* d_mapx, const float* d_mapy,
    const float* d_kernel, int ksz,
    float* d_out, int out_h, int out_w
) {
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y
    );

    get_brightness_kernel << <grid, block >> > (
        d_img, img_h, img_w,
        d_mapx, d_mapy,
        d_kernel, ksz,
        d_out, out_h, out_w
        );

    cudaDeviceSynchronize();
}

// ========== 3. 完整封装接口（C++ 调用）==========
void get_brightness_cuda_wrapper(
    const float* h_img, int img_h, int img_w,
    const float* h_mapx,
    const float* h_mapy,
    const float* h_kernel, int kernel_size,
    float* h_out, int out_h, int out_w
) {
    size_t img_size = img_h * img_w * sizeof(float);
    size_t map_size = out_h * out_w * sizeof(float);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    size_t out_size = out_h * out_w * sizeof(float);

    // 分配 Device 内存
    float* d_img = nullptr;
    float* d_mapx = nullptr;
    float* d_mapy = nullptr;
    float* d_kernel = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_mapx, map_size);
    cudaMalloc(&d_mapy, map_size);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMalloc(&d_out, out_size);

    // Host -> Device 拷贝
    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mapx, h_mapx, map_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mapy, h_mapy, map_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    // 启动核函数
    launch_get_brightness(
        d_img, img_h, img_w,
        d_mapx, d_mapy,
        d_kernel, kernel_size,
        d_out, out_h, out_w
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 出错也记得释放
        cudaFree(d_img);
        cudaFree(d_mapx);
        cudaFree(d_mapy);
        cudaFree(d_kernel);
        cudaFree(d_out);
        return;
    }

    // Device -> Host 拷贝结果
    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

    // 释放 Device 内存
    cudaFree(d_img);
    cudaFree(d_mapx);
    cudaFree(d_mapy);
    cudaFree(d_kernel);
    cudaFree(d_out);
}