#include <mathStructsCPU.h>
#include <iostream>
#include <math.h>

//* MY_VEC
namespace cpu
{
    using namespace std;

    my_vec::my_vec(const vector<DATA_TYPE> &vals) : _size(vals.size()), v(new DATA_TYPE[_size]{0})
    {
        for (int i = 0; i < _size; i++)
            v[i] = vals[i];
    }

    my_vec::my_vec(size_t _size, int mode) : _size(_size), v(new DATA_TYPE[_size]{0})
    {
        if (mode == RANDOM)
            for (int i = 0; i < _size; i++)
                v[i] = DATA_TYPE((float)random() / RAND_MAX * RANGE + net::MIN_RANGE);
    }

    my_vec::my_vec(const my_vec &rh) : _size(rh._size), v(new DATA_TYPE[_size]{0})
    {
        for (int i = 0; i < _size; i++)
            v[i] = rh.v[i];
    }

    my_vec::my_vec(my_vec &&rh) : _size(rh._size), v(move(rh.v)) {}

    my_vec &my_vec::operator=(const my_vec &rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v.reset(new DATA_TYPE[_size]{0});

            for (int i = 0; i < _size; i++)
                v[i] = rh.v[i];
        }

        return *this;
    }

    my_vec &my_vec::operator=(my_vec &&rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = move(rh.v);
        }

        return *this;
    }

    vector<DATA_TYPE> my_vec::copy_inner_vec()
    {
        vector<DATA_TYPE> copy(_size, 0);

        for (int i = 0; i < _size; i++)
            copy[i] = v[i];

        return copy;
    }

    DATA_TYPE my_vec::reduce()
    {
        DATA_TYPE sum = 0;

        for (int i = 0; i < _size; i++)
            sum += v[i];

        return sum;
    }

    my_vec &my_vec::elems_abs()
    {
        for (int i = 0; i < _size; i++)
            v[i] = abs(v[i]);

        return *this;
    }

    DATA_TYPE &my_vec::operator[](int i)
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

        for (int i = 0; i < _size; i++)
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

        for (int i = 0; i < _size; i++)
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
            for (int i = 0; i < _size; i++)
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

        for (int i = 0; i < _size; i++)
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
            for (int i = 0; i < _size; i++)
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

        for (int i = 0; i < _size; i++)
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
            for (int i = 0; i < _size; i++)
                v[i] -= rh.v[i];

        return *this;
    }

    size_t my_vec::size()
    {
        return _size;
    }

    void my_vec::print()
    {
        for (int i = 0; i < _size; i++)
            cout << v[i] << " ";

        cout << "\n";
    }
}

//* MY_VEC_FUN
namespace cpu
{
    using namespace std;

    my_vec_fun::my_vec_fun(size_t _size, bool derivate) : _size(_size)
    {
        v.reserve(_size);

        if (derivate == DERIVATE)
        {
            for (int i = 0; i < _size; i++)
            {
                v.emplace_back(my_vec_fun::relu);    // TODO: implementar bien
                fx.emplace_back(my_vec_fun::fxrelu); // TODO: implementar bien
            }
        }
        else
        {
            for (int i = 0; i < _size; i++)
                v.emplace_back(my_vec_fun::relu); // TODO: implementar bien
        }
    }

    my_vec_fun::my_vec_fun(const vector<size_t> &funs, bool derivate) : _size(_size)
    {
        v.reserve(_size);

        if (derivate == DERIVATE)
        {
            fx.reserve(_size);

            for (int i = 0; i < _size; i++)
            {
                switch (funs[i])
                {
                case RELU:
                case RELU2:
                case SIGMOID:
                default:
                    v.emplace_back(my_vec_fun::relu);
                    fx.emplace_back(my_vec_fun::fxrelu);
                    break;
                }
            }
        }
        else
        {
            for (int i = 0; i < _size; i++)
            {
                switch (funs[i])
                {
                case RELU:
                case RELU2:
                case SIGMOID:
                default:
                    v.emplace_back(my_vec_fun::relu);
                    break;
                }
            }
        }
    }

    my_vec_fun::my_vec_fun(const my_vec_fun &rh) : _size(rh._size), v(rh.v), fx(rh.fx) {}

    my_vec_fun::my_vec_fun(my_vec_fun &&rh) : _size(rh._size), v(move(rh.v)), fx(move(rh.fx)) {}

    my_vec_fun &my_vec_fun::operator=(const my_vec_fun &rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = rh.v;
            fx = rh.fx;
        }

        return *this;
    }

    my_vec_fun &my_vec_fun::operator=(my_vec_fun &&rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = move(rh.v);
            fx = move(rh.fx);
        }

        return *this;
    }

    DATA_TYPE my_vec_fun::relu(DATA_TYPE &in)
    {

        if (in >= 0)
            return in;
        else
            return in / 8;
    }

    DATA_TYPE my_vec_fun::fxrelu(DATA_TYPE &in)
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

        for (int i = 0; i < _size; i++)
            tmp[i] = v[i](rh[i]);

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

        for (int i = 0; i < _size; i++)
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
            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    m[i * _cols + j] = DATA_TYPE((float)random() / RAND_MAX * RANGE + net::MIN_RANGE);
    }

    my_matrix::my_matrix(const vector<vector<DATA_TYPE>> &vecs) : _rows(vecs.size()), _cols(vecs[0].size()), m(new DATA_TYPE[_rows * _cols]{0})
    {
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                m[i * _cols + j] = vecs[i][j];
    }

    my_matrix::my_matrix(const my_matrix &rh) : _rows(rh._rows), _cols(rh._cols), m(new DATA_TYPE[_rows * _cols]{0})
    {
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                m[i * _cols + j] = rh.m[i * _cols + j];
    }

    my_matrix::my_matrix(my_matrix &&rh) : _rows(rh._rows), _cols(rh._cols), m(move(rh.m)) {}

    my_matrix &my_matrix::operator=(const my_matrix &rh)
    {
        if (this != &rh)
        {
            _rows = rh._rows;
            _cols = rh._cols;
            m.reset(new DATA_TYPE[_rows * _cols]{0});

            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    m[i * _cols + j] = rh.m[i * _cols + j];
        }

        return *this;
    }

    my_matrix &my_matrix::operator=(my_matrix &&rh)
    {
        if (this != &rh)
        {
            _rows = rh._rows;
            _cols = rh._cols;
            m = move(rh.m);
        }

        return *this;
    }

    DATA_TYPE &my_matrix::operator()(size_t row, size_t col)
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

        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < rh._cols; j++)
            {
                DATA_TYPE sum = 0;

                for (int k = 0; k < rh._rows; k++)
                    sum += (*this)(i, k) * rh(k, j);

                m[i * _cols + j] = sum;
            }
        }

        return tmp;
    }

    my_matrix my_matrix::operator+(my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(1, 1, CERO);
        }
#endif

        my_matrix tmp(_rows, _cols, CERO);

        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                tmp(i, j) = (*this)(i, j) + rh(i, j);

        return tmp;
    }

    my_matrix &my_matrix::operator+=(my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
        else
#endif
            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    (*this)(i, j) += rh(i, j);

        return *this;
    }

    my_matrix my_matrix::operator-(my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(1, 1, CERO);
        }
#endif

        my_matrix tmp(_rows, _cols, CERO);

        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                tmp(i, j) = (*this)(i, j) - rh(i, j);

        return tmp;
    }

    my_matrix &my_matrix::operator-=(my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
        else
#endif
            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    (*this)(i, j) -= rh(i, j);

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

        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
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
            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    (*this)(i, j) *= rh[i];

        return *this;
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
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
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

        for (int j = 0; j < matrix.cols(); j++)
        {
            DATA_TYPE sum = 0;

            for (int k = 0; k < matrix.rows(); k++)
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

        for (int j = 0; j < matrix.rows(); j++)
        {
            DATA_TYPE sum = 0;

            for (int k = 0; k < matrix.cols(); k++)
                sum += vec[k] * matrix(j, k);

            tmp[j] = sum;
        }

        return tmp;
    }

    my_matrix make_from(my_vec &lh, my_vec &rh)
    {
        my_matrix tmp(lh.size(), rh.size(), CERO);

        for (int i = 0; i < lh.size(); i++)
            for (int j = 0; j < rh.size(); j++)
                tmp(i, j) = lh[i] * rh[j];

        return tmp;
    }
}