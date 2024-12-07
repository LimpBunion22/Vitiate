#ifndef MATHGPU_CUH
#define MATHGPU_CUH

#include <cudaUtils.cuh>
#include <memoryGPU.cuh>

namespace gpu
{
    // common vtor/matrix op
    __device__ float sign_as_unit(float in);
    void sum(float *out, const float *a, const float *b, int size, cudaStream_t stream);
    void substract(float *out, const float *a, const float *b, int size, cudaStream_t stream);
    void sum_inout(float *inout, const float *a, int size, const float *multiplier, cublas_data &cublas, cudaStream_t stream); // device multiplier
    void substract_inout(float *inout, const float *a, int size, float *multiplier, cublas_data &cublas, cudaStream_t stream); // device multiplier
    void divide_by_val_cero_check(float *val, float *data, int size, cudaStream_t stream);                                     // device val
    void divide(float *out, const float *a, const float *b, int size, cudaStream_t stream);
    void multiply(float *out, const float *a, const float *b, int size, cudaStream_t stream);
    void elems_abs(float *data, int size, cudaStream_t stream);
    void reduce(const float *data, int size, cub_data &cub, cudaStream_t stream);
    void set_data(float val, float *data, int size, cudaStream_t stream);
    int set_data(float val, float *data, int size, int roll, cudaStream_t stream);
    void random_set_data(float *data, int size, cudaStream_t stream);
    void random_set_data(float *data, int size, float constant, cudaStream_t stream);

    // specific vtor
    void special_vtor_mul_cero_check(vtor out, const vtor a, const vtor b, cudaStream_t stream);
    void special_vtor_mul_inout(vtor inout, const vtor a, cudaStream_t stream);

    // specific matrix
    void special_matrix_mul(matrix out, const matrix a, const vtor b, cudaStream_t stream);
    void special_matrix_mul_inout_cero_check(matrix inout, const vtor a, cudaStream_t stream);

    // combined op
    void make_matrix_from(matrix out, const vtor a, const vtor b, cudaStream_t stream);
    void matrix_times_vtor_plus_vtor(const matrix a, const vtor b, vtor c_inout, cublas_data &cublas, cudaStream_t stream);
    void vtor_times_matrix(vtor out, const vtor a, const matrix b, cublas_data &cublas, cudaStream_t stream);
}

#endif