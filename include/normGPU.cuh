#ifndef NORMGPU_CUH
#define NORMGPU_CUH

#include <cudaUtils.cuh>
#include <defines.h>

namespace gpu
{
    class abs_max
    {
    public:
        __device__ __forceinline__ float operator()(float a, float b) const
        {
            float abs_a = fabsf(a);
            float abs_b = fabsf(b);
            return (abs_a > abs_b) ? abs_a : abs_b;
        }
    };

    class gradient_norm // for all gradient derivatives
    {
    private:
        float _lambda = 0.1f;
        int _sel_norm = net::DISABLE;
        bool _use_reg = false;
        cub_data *_cub = nullptr;
        cublas_data *_cublas = nullptr;

    private:
        void (gradient_norm::*_norm)(const float *, int, cudaStream_t) = nullptr;
        void (gradient_norm::*_update)(float, int, const float *, const float *, float *, int, cudaStream_t) = &gradient_norm::_sum;
        void _max_norm(const float *data, int size, cudaStream_t stream);
        void _abs_norm(const float *data, int size, cudaStream_t stream);
        void _modulo_norm(const float *data, int size, cudaStream_t stream);

        void _reg(float alpha, int batch_size, const float *norm, const float *fx_accum, float *data, int size, cudaStream_t stream); // regularization
        void _sum(float alpha, int batch_size, const float *norm, const float *fx_accum, float *data, int size, cudaStream_t stream);

    public:
        gradient_norm() = default;
        gradient_norm(const gradient_norm &rh);
        gradient_norm(gradient_norm &&rh);
        gradient_norm &operator=(const gradient_norm &rh);
        gradient_norm &operator=(gradient_norm &&rh);
        ~gradient_norm() = default;

        void select_norm(int norm);
        void use_reg(bool use);
        void set_lambda(float lambda);
        void set_cub(cub_data &cub);
        void set_cublas(cublas_data &cublas);
        void norm(float alpha, const float *fx_accum, float *data, int size, int batch_size, cudaStream_t stream);
    };
}

#endif