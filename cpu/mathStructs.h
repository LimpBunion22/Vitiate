#ifndef MATHSTRUCTS_H
#define MATHSTRUCTS_H

#include <defines.h>
#include <vector>

namespace cpu
{
    constexpr long RANGE = 2 * MAX_RANGE;
    constexpr bool DERIVATE = true;
    constexpr bool NOT_DERIVATE = false;
    constexpr int RELU = 0;
    constexpr int RELU2 = 1;
    constexpr int SIGMOID = 2;
    constexpr int RANDOM = 1;
    constexpr int CERO = 2;

    using my_fun = DATA_TYPE (*)(DATA_TYPE &in);

    class my_vec
    {
    private:
        size_t _size;
        std::vector<DATA_TYPE> v;

    private:
        my_vec() = delete;

    public:
        my_vec(std::vector<DATA_TYPE> &vals);
        my_vec(size_t _size, int mode);
        my_vec(std::initializer_list<DATA_TYPE> l);
        my_vec(const my_vec &rh);
        my_vec(my_vec &&rh);
        my_vec &operator=(const my_vec &rh);
        my_vec &operator=(my_vec &&rh);

        std::vector<DATA_TYPE> copy_inner_vec();
        DATA_TYPE reduce();
        my_vec &elems_abs();

        DATA_TYPE &operator[](int i);
        DATA_TYPE operator*(const my_vec &rh);
        my_vec operator^(const my_vec &rh);
        my_vec &operator^=(const my_vec &rh);
        my_vec operator+(const my_vec &rh);
        my_vec &operator+=(const my_vec &rh);
        my_vec operator-(const my_vec &rh);
        my_vec &operator-=(const my_vec &rh);

        size_t size();
        void print();
    };

    class my_vec_fun
    {
    private:
        size_t _size;
        std::vector<my_fun> v;
        std::vector<my_fun> fx;

    private:
        my_vec_fun() = delete;

    public:
        my_vec_fun(size_t _size, bool derivate);
        my_vec_fun(std::vector<size_t> &funs, bool derivate);
        my_vec_fun(const my_vec_fun &rh);
        my_vec_fun(my_vec_fun &&rh);
        my_vec_fun &operator=(const my_vec_fun &rh);
        my_vec_fun &operator=(my_vec_fun &&rh);

        static DATA_TYPE relu(DATA_TYPE &in);
        static DATA_TYPE fxrelu(DATA_TYPE &in);
        my_vec calculate(my_vec &rh);
        my_vec derivate(my_vec &rh);

        size_t size();
    };

    class my_matrix
    {
        friend my_matrix make_from(my_vec &lh, my_vec &rh);

    private:
        size_t _rows;
        size_t _cols;
        std::vector<my_vec> m;

    private:
        my_matrix() = delete;
        my_matrix(size_t _rows, size_t _cols);

    public:
        my_matrix(size_t _rows, size_t _cols, int mode);
        my_matrix(std::vector<std::vector<DATA_TYPE>> &vecs);
        my_matrix(std::initializer_list<std::initializer_list<DATA_TYPE>> l);
        my_matrix(const my_matrix &rh);
        my_matrix(my_matrix &&rh);
        my_matrix &operator=(const my_matrix &rh);
        my_matrix &operator=(my_matrix &&rh);

        my_vec &operator[](size_t row);
        my_matrix operator*(my_matrix &rh);
        my_matrix operator+(my_matrix &rh);
        my_matrix &operator+=(my_matrix &rh);
        my_matrix operator-(my_matrix &rh);
        my_matrix &operator-=(my_matrix &rh);
        my_matrix operator^(my_vec &rh);
        my_matrix &operator^=(my_vec &rh);

        size_t rows();
        size_t cols();
        void print();
    };

    //* friend functions
    my_vec operator*(my_vec &vec, my_matrix &matrix);
    my_vec operator*(my_matrix &matrix, my_vec &vec);
    my_matrix make_from(my_vec &lh, my_vec &rh);
}
#endif
