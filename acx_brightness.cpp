#include <opencv2/opencv.hpp>
#include <immintrin.h>
#include <omp.h>

// 内部 5×5 卷积，返回一个点结果
static inline float avx2_conv5x5(const float* ptr, int stride, const float* k) {
    __m256 v[5], kv[5];
    for (int i = 0; i < 5; ++i) {
        v[i] = _mm256_loadu_ps(ptr + i * stride);
        kv[i] = _mm256_loadu_ps(k + i * 5);
    }
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < 5; ++i) sum = _mm256_fmadd_ps(v[i], kv[i], sum);
    float buf[8];
    _mm256_storeu_ps(buf, sum);
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] +
        buf[5] + buf[6] + buf[7];
}

void _get_brightness(cv::Mat img,
    cv::Mat mapx,
    cv::Mat mapy,
    cv::Mat kernel,
    cv::Mat& csv)
{
    CV_Assert(img.type() == CV_32FC1);
    CV_Assert(mapx.type() == CV_32FC1 && mapy.type() == CV_32FC1);
    CV_Assert(kernel.type() == CV_32FC1 && kernel.rows == 5 && kernel.cols == 5);
    csv.create(mapx.size(), CV_32FC1);

    const int rad = 2;
    const int ksz = 5;
    float k[25];
    for (int i = 0; i < 25; ++i) k[i] = kernel.ptr<float>()[i];

#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < mapx.rows; ++i) {
        const float* mx = mapx.ptr<float>(i);
        const float* my = mapy.ptr<float>(i);
        float* out = csv.ptr<float>(i);
        for (int j = 0; j < mapx.cols; ++j) {
            int x = static_cast<int>(mx[j]);
            int y = static_cast<int>(my[j]);
            if (x < rad || y < rad || x >= img.rows - rad || y >= img.cols - rad) {
                out[j] = 0.0f;
                continue;
            }
            const float* base = img.ptr<float>(x - rad) + (y - rad);
            float sum = 0.0f;
            for (int dy = 0; dy < ksz; ++dy)
                for (int dx = 0; dx < ksz; ++dx)
                    sum += base[dy * img.cols + dx] * k[dy * ksz + dx];
            out[j] = sum;
        }
    }
}