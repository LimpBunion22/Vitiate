#include <mathStructs.h>
#include <iostream>
#include <math.h>

//* MY_VEC
namespace cpu
{
    using namespace std;

    my_vec::my_vec(vector<DATA_TYPE> &vals) : _size(vals.size())
    {
        v = move(vals);
    }

    my_vec::my_vec(size_t _size, int mode) : _size(_size), v(_size, 0)
    {
        if (mode == RANDOM)
            for (int i = 0; i < _size; i++)
                v[i] = DATA_TYPE((float)random() / RAND_MAX * RANGE + MIN_RANGE);
    }

    my_vec::my_vec(initializer_list<DATA_TYPE> l) : v(l), _size(l.size()) {}

    my_vec::my_vec(const my_vec &rh) : _size(rh._size), v(rh.v) {}

    my_vec::my_vec(my_vec &&rh) : _size(rh._size), v(move(rh.v)) {}

    my_vec &my_vec::operator=(const my_vec &rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = rh.v;
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
        return v;
    }

    DATA_TYPE my_vec::reduce()
    {
        DATA_TYPE sum = 0;

        for (auto &i : v)
            sum += i;

        return sum;
    }

    my_vec &my_vec::elems_abs()
    {
        for (auto &i : v)
            i = abs(i);

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
            return my_vec(0, CERO);
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
            return my_vec(0, CERO);
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
            return my_vec(0, CERO);
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
        for (auto &el : v)
            cout << el << " ";

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

    my_vec_fun::my_vec_fun(vector<size_t> &funs, bool derivate) : _size(_size)
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
            return my_vec(0, CERO);
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
            return my_vec(0, CERO);
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

    my_matrix::my_matrix(size_t _rows, size_t _cols) : _rows(_rows), _cols(_cols)
    {
        m.reserve(_rows);
    }

    my_matrix::my_matrix(size_t _rows, size_t _cols, int mode) : _rows(_rows), _cols(_cols), m(_rows, my_vec(_cols, CERO))
    {

        if (mode == RANDOM)
            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    m[i][j] = DATA_TYPE((float)random() / RAND_MAX * RANGE + MIN_RANGE);
    }

    my_matrix::my_matrix(vector<vector<DATA_TYPE>> &vecs) : _rows(vecs.size()), _cols(vecs[0].size())
    {
        m.reserve(_rows);

        for (int i = 0; i < _rows; i++)
            m.emplace_back(vecs[i]);
    }

    my_matrix::my_matrix(initializer_list<initializer_list<DATA_TYPE>> l) : _rows(l.size()), _cols(l.begin()->size())
    {
        m.reserve(l.size());

        for (auto &list : l)
            m.emplace_back(list);
    }

    my_matrix::my_matrix(const my_matrix &rh) : _rows(rh._rows), _cols(rh._cols), m(rh.m) {}

    my_matrix::my_matrix(my_matrix &&rh) : _rows(rh._rows), _cols(rh._cols), m(move(rh.m)) {}

    my_matrix &my_matrix::operator=(const my_matrix &rh)
    {
        if (this != &rh)
        {
            _rows = rh._rows;
            _cols = rh._cols;
            m = rh.m;
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

    my_vec &my_matrix::operator[](size_t row)
    {
#ifdef ASSERT
        if (row < _rows)
            return m[row];

        cout << "invalid access\n";
        exit(EXIT_FAILURE);
#else
        return m[row];
#endif
    }

    my_matrix my_matrix::operator*(my_matrix &rh)
    {
#ifdef ASSERT
        if (_cols != rh._rows)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(0, 0);
        }
#endif

        my_matrix tmp(_rows, rh._cols);

        for (int i = 0; i < _rows; i++)
        {
            my_vec row(rh._cols, CERO);

            for (int j = 0; j < rh._cols; j++)
            {
                DATA_TYPE sum = 0;

                for (int k = 0; k < rh._rows; k++)
                    sum += (*this)[i][k] * rh[k][j];

                row[i] = sum;
            }

            tmp.m.emplace_back(row);
        }

        return tmp;
    }

    my_matrix my_matrix::operator+(my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(0, 0);
        }
#endif

        my_matrix tmp(_rows, _cols);

        for (int i = 0; i < _rows; i++)
            tmp.m.emplace_back((*this)[i] + rh[i]);

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
                (*this)[i] += rh[i];

        return *this;
    }

    my_matrix my_matrix::operator-(my_matrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << "\n";
            return my_matrix(0, 0);
        }
#endif

        my_matrix tmp(_rows, _cols);

        for (int i = 0; i < _rows; i++)
            tmp.m.emplace_back((*this)[i] - rh[i]);

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
                (*this)[i] -= rh[i];

        return *this;
    }

    my_matrix my_matrix::operator^(my_vec &rh)
    {
#ifdef ASSERT
        if (rh.size() != _rows)
        {
            cout << "invalid dimensions lh is " << _rows << " rh is " << rh.size() << "\n";
            return my_matrix(0, 0);
        }
#endif

        my_matrix tmp = *this;

        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                tmp[i][j] *= rh[i];

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
                    (*this)[i][j] *= rh[i];

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
        for (auto &row : m)
            row.print();
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
            return my_vec(0, CERO);
        }
#endif

        my_vec tmp(matrix.cols(), CERO);

        for (int j = 0; j < matrix.cols(); j++)
        {
            DATA_TYPE sum = 0;

            for (int k = 0; k < matrix.rows(); k++)
                sum += vec[k] * matrix[k][j];

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
            return my_vec(0, CERO);
        }
#endif

        my_vec tmp(matrix.rows(), CERO);

        for (int j = 0; j < matrix.rows(); j++)
        {
            DATA_TYPE sum = 0;

            for (int k = 0; k < matrix.cols(); k++)
                sum += vec[k] * matrix[j][k];

            tmp[j] = sum;
        }

        return tmp;
    }

    my_matrix make_from(my_vec &lh, my_vec &rh)
    {
        my_matrix tmp(lh.size(), rh.size());

        for (int i = 0; i < lh.size(); i++)
        {
            my_vec row(rh.size(), CERO);

            for (int j = 0; j < rh.size(); j++)
                row[j] = lh[i] * rh[j];

            tmp.m.emplace_back(row);
        }

        return tmp;
    }
}