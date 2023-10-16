#ifndef MATHCPU_H
#define MATHCPU_H

#include <memoryCPU.h>

namespace cpu::single
{
    // common vtor/matrix op
    float sign_as_unit(float in);
    void sum(float *__restrict out, const float *__restrict a, const float *__restrict b, int size);
    void substract(float *__restrict out, const float *__restrict a, const float *__restrict b, int size);
    void sum_inout(float *__restrict inout, const float *__restrict a, int size, float multiplier);
    void substract_inout(float *__restrict inout, const float *__restrict a, int size, float multiplier);
    void divide_by_val_cero_check(float val, float *__restrict data, int size);
    void elems_abs(float *__restrict data, int size);
    float reduce(const float *__restrict data, int size);
    void set_data(float val, float *__restrict data, int size);
    void random_set_data(float *__restrict data, int size);
    void random_set_data(float *__restrict data, int size, float constant);
    void copy_data(float *__restrict dst, const float *__restrict in, int size);

    // specific vtor
    void special_vtor_mul_cero_check(vtor out, const vtor a, const vtor b);
    void special_vtor_mul_inout(vtor inout, const vtor a);
    float vtor_mul(const vtor a, const vtor b);

    // specific matrix
    void special_matrix_mul(matrix out, const matrix a, const vtor b);
    void special_matrix_mul_inout_cero_check(matrix inout, const vtor a);
    void matrix_mul(matrix out, const matrix a, const matrix b);

    // combined op
    void make_matrix_from(matrix out, const vtor a, const vtor b);
    void vtor_times_matrix(vtor out, const vtor a, const matrix b);
}

#endif