#ifndef MATHSTRUCTSCPU_H
#define MATHSTRUCTSCPU_H

#include <defines.h>
#include <memory>

namespace cpu
{
    constexpr long RANGE = 2 * net::MAX_RANGE;
    constexpr bool DERIVATE = true;
    constexpr bool NOT_DERIVATE = false;
    constexpr int RELU = 0;
    constexpr int RELU2 = 1;
    constexpr int SIGMOID = 2;
    constexpr int RANDOM = 1;
    constexpr int CERO = 2;

    using my_fun = DATA_TYPE (*)(const DATA_TYPE &in);

    class my_vec
    {
    private:
        size_t _size;
        DATA_TYPE *v;

    private:
        my_vec() = delete;

    public:
        my_vec(const std::vector<DATA_TYPE> &vals);
        my_vec(size_t _size, int mode);
        my_vec(const my_vec &rh);
        my_vec(my_vec &&rh);
        ~my_vec();
        my_vec &operator=(const my_vec &rh);
        my_vec &operator=(my_vec &&rh);

        std::vector<DATA_TYPE> copy_inner_vec();
        DATA_TYPE reduce();
        my_vec &elems_abs();
        void reset();
        void divide_elems_by(const DATA_TYPE &val);

        DATA_TYPE &operator[](const size_t &i);
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
        my_fun *f;
        my_fun *fx;

    private:
        my_vec_fun() = delete;

    public:
        my_vec_fun(size_t _size, bool derivate);
        my_vec_fun(const std::vector<size_t> &funs, bool derivate);
        my_vec_fun(const my_vec_fun &rh);
        my_vec_fun(my_vec_fun &&rh);
        my_vec_fun &operator=(const my_vec_fun &rh);
        my_vec_fun &operator=(my_vec_fun &&rh);
        ~my_vec_fun();

        static DATA_TYPE relu(const DATA_TYPE &in);
        static DATA_TYPE fxrelu(const DATA_TYPE &in);
        my_vec calculate(my_vec &rh);
        my_vec derivate(my_vec &rh);

        size_t size();
    };

    class my_matrix
    {
    private:
        size_t _rows;
        size_t _cols;
        DATA_TYPE *m;

    private:
        my_matrix() = delete;

    public:
        my_matrix(size_t _rows, size_t _cols, int mode);
        my_matrix(const std::vector<std::vector<DATA_TYPE>> &vecs);
        my_matrix(const my_matrix &rh);
        my_matrix(my_matrix &&rh);
        my_matrix &operator=(const my_matrix &rh);
        my_matrix &operator=(my_matrix &&rh);
        ~my_matrix();

        DATA_TYPE &operator()(const size_t &row, const size_t &col);
        my_matrix operator*(my_matrix &rh);
        my_matrix operator+(const my_matrix &rh);
        my_matrix &operator+=(const my_matrix &rh);
        my_matrix operator-(const my_matrix &rh);
        my_matrix &operator-=(const my_matrix &rh);
        my_matrix operator^(my_vec &rh);
        my_matrix &operator^=(my_vec &rh);
        void reset();
        void divide_elems_by(const DATA_TYPE &val);

        size_t rows();
        size_t cols();
        void print();
    };

    //*functions
    my_vec operator*(my_vec &vec, my_matrix &matrix);
    my_vec operator*(my_matrix &matrix, my_vec &vec);
    my_matrix make_from(my_vec &lh, my_vec &rh);
}
#endif
