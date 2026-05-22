#ifndef GET_BRIGHTNESS_CUDA_CUH
#define GET_BRIGHTNESS_CUDA_CUH

#ifdef __cplusplus
extern "C" {
#endif

    // C++ 调用的顶层接口：包含内存分配、数据传输、核函数启动、结果拷回
    void get_brightness_cuda_wrapper(
        const float* h_img, int img_h, int img_w,
        const float* h_mapx,
        const float* h_mapy,
        const float* h_kernel, int kernel_size,
        float* h_out, int out_h, int out_w
    );

#ifdef __cplusplus
}
#endif

#endif