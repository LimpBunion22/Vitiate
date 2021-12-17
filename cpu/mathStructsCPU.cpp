#include <mathStructsCPU.h>
#include <iostream>
#include <math.h>

//* MY_VEC
namespace cpu
{
    using namespace std;

    my_vec::my_vec(const vector<DATA_TYPE> &vals) : _size(vals.size()), v(new DATA_TYPE[_size])
    {
        for (size_t i = 0; i < _size; i++)
            v[i] = vals[i];
    }

    my_vec::my_vec(size_t _size, int mode) : _size(_size), v(new DATA_TYPE[_size]{0})
    {
        if (mode == RANDOM)
            for (size_t i = 0; i < _size; i++)
                v[i] = DATA_TYPE((float)random() / RAND_MAX * RANGE + net::MIN_RANGE);
    }

    my_vec::my_vec(const my_vec &rh) : _size(rh._size), v(new DATA_TYPE[_size])
    {
        for (size_t i = 0; i < _size; i++)
            v[i] = rh.v[i];
    }

    my_vec::my_vec(my_vec &&rh) : _size(rh._size)
    {
        v = rh.v;
        rh.v = nullptr;
    }

    my_vec &my_vec::operator=(const my_vec &rh)
    {
        if (this != &rh)
        {
            if (_size != rh._size)
            {
                _size = rh._size;
                delete[] v;
                v = new DATA_TYPE[_size];
            }

            for (size_t i = 0; i < _size; i++)
                v[i] = rh.v[i];
        }

        return *this;
    }

    my_vec &my_vec::operator=(my_vec &&rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            delete[] v;
            v = rh.v;
            rh.v = nullptr;
        }

        return *this;
    }

    my_vec::~my_vec()
    {
        delete[] v;
    }

    vector<DATA_TYPE> my_vec::copy_inner_vec()
    {
        vector<DATA_TYPE> copy(_size, 0);

        for (size_t i = 0; i < _size; i++)
            copy[i] = v[i];

        return copy;
    }

    DATA_TYPE my_vec::reduce()
    {
        DATA_TYPE sum = 0;

        for (size_t i = 0; i < _size; i++)
            sum += v[i];

        return sum;
    }

    my_vec &my_vec::elems_abs()
    {
        for (size_t i = 0; i < _size; i++)
            v[i] = abs(v[i]);

        return *this;
    }

    void my_vec::reset()
    {
        for (size_t i = 0; i < _size; i++)
            v[i] = 0;
    }

    void my_vec::divide_elems_by(const DATA_TYPE &val)
    {
        for (size_t i = 0; i < _size; i++)
            v[i] /= val;
    }

    DATA_TYPE &my_vec::operator[](const size_t &i)
    {
#ifdef ASSERT
        if (i < _size)
            return v[i];

        cout << "invalid access\n";
        exit(EXIT_FAILURE);
#else
        return v[i];
#endif
    }

    DATA_TYPE my_vec::operator*(const my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << "\n";
            return 0;
        }
#endif

        DATA_TYPE sum = 0;

        for (size_t i = 0; i < _size; i++)
            sum += v[i] * rh.v[i];

        return sum;
    }

    my_vec my_vec::operator^(const my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << "\n";
            return my_vec(1, CERO);
        }
#endif

        my_vec tmp(_size, CERO);

        for (size_t i = 0; i < _size; i++)
            tmp.v[i] = v[i] * rh.v[i];

        return tmp;
    }

    my_vec &my_vec::operator^=(const my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << "\n";
        else
#endif
            for (size_t i = 0; i < _size; i++)
                v[i] *= rh.v[i];

        return *this;
    }

    my_vec my_vec::operator+(const my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << "\n";
            return my_vec(1, CERO);
        }
#endif

        my_vec tmp(_size, CERO);

        for (size_t i = 0; i < _size; i++)
            tmp.v[i] = v[i] + rh.v[i];

        return tmp;
    }

    my_vec &my_vec::operator+=(const my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << "\n";
        else
#endif
            for (size_t i = 0; i < _size; i++)
                v[i] += rh.v[i];

        return *this;
    }

    my_vec my_vec::operator-(const my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << "\n";
            return my_vec(1, CERO);
        }
#endif

        my_vec tmp(_size, CERO);

        for (size_t i = 0; i < _size; i++)
            tmp.v[i] = v[i] - rh.v[i];

        return tmp;
    }

    my_vec &my_vec::operator-=(const my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << "\n";
        else
#endif
            for (size_t i = 0; i < _size; i++)
                v[i] -= rh.v[i];

        return *this;
    }

    size_t my_vec::size()
    {
        return _size;
    }

    void my_vec::print()
    {
        for (size_t i = 0; i < _size; i++)
            cout << v[i] << " ";

        cout << "\n";
    }
}

//* MY_VEC_FUN
namespace cpu
{
    using namespace std;

    my_vec_fun::my_vec_fun(size_t _size, bool derivate) : _size(_size), f(new my_fun[_size])
    {
        if (derivate == DERIVATE)
        {
            fx = new my_fun[_size];

            for (size_t i = 0; i < _size; i++)
            {
                f[i] = my_vec_fun::relu;    // TODO: implementar bien
                fx[i] = my_vec_fun::fxrelu; // TODO: implementar bien
            }
        }
        else
        {
            fx = nullptr;

            for (size_t i = 0; i < _size; i++)
                f[i] = my_vec_fun::relu; // TODO: implementar bien
        }
    }

    my_vec_fun::my_vec_fun(const vector<size_t> &funs, bool derivate) : _size(_size), f(new my_fun[_size])
    {
        if (derivate == DERIVATE)
        {
            fx = new my_fun[_size];

            for (size_t i = 0; i < _size; i++)
            {
                switch (funs[i])
                {
                case RELU:
                case RELU2:
                case SIGMOID:
                default:
                    f[i] = my_vec_fun::relu;    // TODO: implementar bien
                    fx[i] = my_vec_fun::fxrelu; // TODO: implementar bien
                    break;
                }
            }
        }
        else
        {
            fx = nullptr;

            for (size_t i = 0; i < _size; i++)
            {
                switch (funs[i])
                {
                case RELU:
                case RELU2:
                case SIGMOID:
                default:
                    f[i] = my_vec_fun::relu; // TODO: implementar bien
                    break;
                }
            }
        }
    }

    my_vec_fun::my_vec_fun(const my_vec_fun &rh) : _size(rh._size), f(new my_fun[_size])
    {
        if (rh.fx)
        {
            fx = new my_fun[_size];

            for (size_t i = 0; i < _size; i++)
            {
                f[i] = rh.f[i];
                fx[i] = rh.fx[i];
            }
        }
        else
        {
            fx = nullptr;

            for (size_t i = 0; i < _size; i++)
                f[i] = rh.f[i];
        }
    }

    my_vec_fun::my_vec_fun(my_vec_fun &&rh) : _size(rh._size)
    {
        fx = rh.fx;
        rh.fx = nullptr;
        f = rh.f;
        rh.f = nullptr;
    }

    my_vec_fun &my_vec_fun::operator=(const my_vec_fun &rh)
    {
        if (this != &rh)
        {
            if (fx && rh.fx)
            {
                if (_size != rh._size)
                {
                    delete[] fx;
                    fx = new my_fun[rh._size];
                }
            }
            else if (fx && !rh.fx)
            {
                delete[] fx;
                fx = nullptr;
            }
            else if (!fx && rh.fx)
                fx = new my_fun[rh._size];

            if (_size != rh._size)
            {
                _size = rh._size;
                delete[] f;
                f = new my_fun[_size];
            }

            if (fx)
                for (size_t i = 0; i < _size; i++)
                {
                    f[i] = rh.f[i];
                    fx[i] = rh.fx[i];
                }
            else
                for (size_t i = 0; i < _size; i++)
                    f[i] = rh.f[i];
        }

        return *this;
    }

    my_vec_fun &my_vec_fun::operator=(my_vec_fun &&rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            delete[] f;
            f = rh.f;
            rh.f = nullptr;
            delete[] fx;
            fx = rh.fx;
            rh.fx = nullptr;
        }

        return *this;
    }

    my_vec_fun::~my_vec_fun()
    {
        delete[] f;
        delete[] fx;
    }

    DATA_TYPE my_vec_fun::relu(const DATA_TYPE &in)
    {

        if (in >= 0)
            return in;
        else
            return in / 8;
    }

    DATA_TYPE my_vec_fun::fxrelu(const DATA_TYPE &in)
    {

        if (in >= 0)
            return 1;
        else
            return (DATA_TYPE)1 / 8;
    }

    my_vec my_vec_fun::calculate(my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh.size())
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh.size() << "\n";
            return my_vec(1, CERO);
        }
#endif

        my_vec tmp(_size, CERO);

        for (size_t i = 0; i < _size; i++)
            tmp[i] = f[i](rh[i]);

        return tmp;
    }

    my_vec my_vec_fun::derivate(my_vec &rh)
    {
#ifdef ASSERT
        if (_size != rh.size())
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh.size() << "\n";
            return my_vec(1, CERO);
        }
#endif

        my_vec tmp(_size, CERO);

        for (size_t i = 0; i < _size; i++)
            tmp[i] = fx[i](rh[i]);

        return tmp;
    }

    size_t my_vec_fun::size()
    {
        return _size;
    }
}

//* MY_MATRIX
namespace cpu
{
    using namespace std;

    my_matrix::my_matrix(size_t _rows, size_t _cols, int mode) : _rows(_rows), _cols(_cols), m(new DATA_TYPE[_rows * _cols]{0})
    {
        if (mode == RANDOM)
            for (size_t i = 0; i < _rows * _cols; i++)
                m[i] = DATA_TYPE((float)random() / RAND_MAX * RANGE + net::MIN_RANGE);
    }

    my_matrix::my_matrix(const vector<vector<DATA_TYPE>> &vecs) : _rows(vecs.size()), _cols(vecs[0].size()), m(new DATA_TYPE[_rows * _cols])
    {
        for (size_t i = 0; i < _rows; i++)
            for (size_t j = 0; j < _cols; j++)
                m[i * _cols + j] = vecs[i][j];
    }

    my_matrix::my_matrix(const my_matrix &rh) : _rows(rh._rows), _cols(rh._cols), m(new DATA_TYPE[_rows * _cols])
    {
        for (size_t i = 0; i < _rows * _cols; i++)
            m[i] = rh.m[i];
    }

    my_matrix::my_matrix(my_matrix &&rh) : _rows(rh._rows), _cols(rh._cols)
    {
        m = rh.m;
        rh.m = nullptr;
    }

    my_matrix &my_matrix::operator=(const my_matrix &rh)
    {
        if (this != &rh)
        {
            if (_rows != rh._rows || _cols != rh._cols)
            {
                _rows = rh._rows;
                _cols = rh._cols;
                delete[] m;
                m = new DATA_TYPE[_rows * _cols];
            }
            for (size_t i = 0; i < _rows * _cols; i++)
                m[i] = rh.m[i];
        }

        return *this;
    }

    my_matrix &my_matrix::operator=(my_matrix &&rh)
    {

        if (this != &rh)
        {
            _rows = rh._rows;
            _cols = rh._cols;
            delete[] m;
            m = rh.m;
            rh.m = nullptr;
        }

        return *this;
    }

    my_matrix::~my_matrix()
    {
        delete[] m;
    }

    DATA_TYPE &my_matrix::operator()(const size_t &row, const size_t &col)
    {
#ifdef ASSERT
        if (row < _rows && col < _cols)
            return m[row * _cols + col];

        cout << "invalid access\n";
        exit(EXIT_FAILURE);
#else
        return m[row * _cols + col];
#endif
    }

    my_matrix my_matrix::operator*(my_matrix &rh)
    {
#ifdef ASSERT
        if (_cols != rh._rows)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(1, 1, CERO);
        }
#endif

        my_matrix tmp(_rows, rh._cols, CERO);

        for (size_t i = 0; i < _rows; i++)
        {
            for (size_t j = 0; j < rh._cols; j++)
            {
                DATA_TYPE sum = 0;

                for (size_t k = 0; k < rh._rows; k++)
                    sum += (*this)(i, k) * rh(k, j);

                m[i * _cols + j] = sum;
            }
        }

        return tmp;
    }

    my_matrix my_matrix::operator+(const my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(1, 1, CERO);
        }
#endif

        my_matrix tmp(_rows, _cols, CERO);

        for (size_t i = 0; i < _rows * _cols; i++)
            tmp.m[i] = m[i] + rh.m[i];

        return tmp;
    }

    my_matrix &my_matrix::operator+=(const my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
        else
#endif
            for (size_t i = 0; i < _rows * _cols; i++)
                m[i] += rh.m[i];

        return *this;
    }

    my_matrix my_matrix::operator-(const my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(1, 1, CERO);
        }
#endif

        my_matrix tmp(_rows, _cols, CERO);

        for (size_t i = 0; i < _rows * _cols; i++)
            tmp.m[i] = m[i] - rh.m[i];

        return tmp;
    }

    my_matrix &my_matrix::operator-=(const my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
        else
#endif
            for (size_t i = 0; i < _rows * _cols; i++)
                m[i] -= rh.m[i];

        return *this;
    }

    my_matrix my_matrix::operator^(my_vec &rh)
    {
#ifdef ASSERT
        if (rh.size() != _rows)
        {
            cout << "invalid dimensions lh is " << _rows << " rh is " << rh.size() << "\n";
            return my_matrix(1, 1, CERO);
        }
#endif

        my_matrix tmp(_rows, _cols, CERO);

        for (size_t i = 0; i < _rows; i++)
            for (size_t j = 0; j < _cols; j++)
                tmp(i, j) = (*this)(i, j) * rh[i];

        return tmp;
    }

    my_matrix &my_matrix::operator^=(my_vec &rh)
    {
#ifdef ASSERT
        if (rh.size() != _rows)
            cout << "invalid dimensions lh is " << _rows << " rh is " << rh.size() << "\n";
        else
#endif
            for (size_t i = 0; i < _rows; i++)
                for (size_t j = 0; j < _cols; j++)
                    (*this)(i, j) *= rh[i];

        return *this;
    }

    void my_matrix::reset()
    {
        for (size_t i = 0; i < _rows * _cols; i++)
            m[i] = 0;
    }

    void my_matrix::divide_elems_by(const DATA_TYPE &val)
    {
        for (size_t i = 0; i < _rows * _cols; i++)
            m[i] /= val;
    }

    size_t my_matrix::rows()
    {
        return _rows;
    }

    size_t my_matrix::cols()
    {
        return _cols;
    }

    void my_matrix::print()
    {
        for (size_t i = 0; i < _rows; i++)
        {
            for (size_t j = 0; j < _cols; j++)
                cout << m[i * _cols + j] << " ";

            cout << "\n";
        }

        cout << "\n";
    }
}

//*friend functions
namespace cpu
{
    using namespace std;

    my_vec operator*(my_vec &vec, my_matrix &matrix)
    {
#ifdef ASSERT
        if (vec.size() != matrix.rows())
        {
            cout << "invalid dimensions lh is " << vec.size() << " rh is " << matrix.rows() << "\n";
            return my_vec(1, CERO);
        }
#endif

        my_vec tmp(matrix.cols(), CERO);
        size_t m_cols = matrix.cols();
        size_t m_rows = matrix.rows();

        for (size_t j = 0; j < m_cols; j++)
        {
            DATA_TYPE sum = 0;

            for (size_t k = 0; k < m_rows; k++)
                sum += vec[k] * matrix(k, j);

            tmp[j] = sum;
        }

        return tmp;
    }

    my_vec operator*(my_matrix &matrix, my_vec &vec)
    {
#ifdef ASSERT
        if (matrix.cols() != vec.size())
        {
            cout << "invalid dimensions lh is " << matrix.cols() << " rh is " << vec.size() << "\n";
            return my_vec(1, CERO);
        }
#endif

        my_vec tmp(matrix.rows(), CERO);
        size_t m_cols = matrix.cols();
        size_t m_rows = matrix.rows();

        for (size_t j = 0; j < m_rows; j++)
        {
            DATA_TYPE sum = 0;

            for (size_t k = 0; k < m_cols; k++)
                sum += vec[k] * matrix(j, k);

            tmp[j] = sum;
        }

        return tmp;
    }

    my_matrix make_from(my_vec &lh, my_vec &rh)
    {
        my_matrix tmp(lh.size(), rh.size(), CERO);
        size_t lh_size = lh.size();
        size_t rh_size = rh.size();

        for (size_t i = 0; i < lh_size; i++)
            for (size_t j = 0; j < rh_size; j++)
                tmp(i, j) = lh[i] * rh[j];

        return tmp;
    }
}