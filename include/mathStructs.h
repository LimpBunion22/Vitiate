#ifndef MATHSTRUCTS_H
#define MATHSTRUCTS_H

#include <defines.h>
#include <vector>
#include <iostream>

constexpr long RANGE = 2 * MAX_RANGE;
constexpr bool DERIVATE = true;
constexpr bool NOT_DERIVATE = false;
constexpr int RELU = 0;
constexpr int RELU2 = 1;
constexpr int SIGMOID = 2;
constexpr int RANDOM = 1;
constexpr int CERO = 2;

using namespace std;

template <class U>
using myFun = U (*)(U &in);

template <class T>
class myMatrix;

template <class T>
class myVec
{
private:
    size_t _size;
    vector<T> v;

private:
    myVec() = delete;

public:
    myVec(vector<T> &vals) : _size(vals.size())
    {
        v = move(vals);
    }

    myVec(size_t _size, int mode) : _size(_size), v(_size, 0)
    {
        if (mode == RANDOM)
            for (int i = 0; i < _size; i++)
                v[i] = T((float)random() / RAND_MAX * RANGE + MIN_RANGE);
    }

    myVec(initializer_list<T> l) : v(l), _size(l.size()) {}

    myVec(const myVec &rh) : _size(rh._size), v(rh.v) {}

    myVec(myVec &&rh) : _size(rh._size), v(move(rh.v)) {}

    myVec &operator=(const myVec &rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = rh.v;
        }

        return *this;
    }

    myVec &operator=(myVec &&rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = move(rh.v);
        }

        return *this;
    }

    T &operator[](int i)
    {
#ifdef ASSERT
        if (i < _size)
            return v[i];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
#else
        return v[i];
#endif
    }

    T operator*(const myVec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return 0;
        }
#endif

        T sum = 0;

        for (int i = 0; i < _size; i++)
            sum += v[i] * rh.v[i];

        return sum;
    }

    myVec operator^(const myVec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec(0, CERO);
        }
#endif

        myVec<T> tmp(_size, CERO);

        for (int i = 0; i < _size; i++)
            tmp.v[i] = v[i] * rh.v[i];

        return tmp;
    }

    myVec &operator^=(const myVec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
        else
#endif
            for (int i = 0; i < _size; i++)
                v[i] *= rh.v[i];

        return *this;
    }

    myVec operator+(const myVec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec(0);
        }
#endif

        myVec<T> tmp(_size, CERO);

        for (int i = 0; i < _size; i++)
            tmp.v[i] = v[i] + rh.v[i];

        return tmp;
    }

    myVec &operator+=(const myVec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
        else
#endif
            for (int i = 0; i < _size; i++)
                v[i] += rh.v[i];

        return *this;
    }

    myVec operator-(const myVec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec(0, CERO);
        }
#endif

        myVec<T> tmp(_size, CERO);

        for (int i = 0; i < _size; i++)
            tmp.v[i] = v[i] - rh.v[i];

        return tmp;
    }

    myVec &operator-=(const myVec &rh)
    {
#ifdef ASSERT
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
        else
#endif
            for (int i = 0; i < _size; i++)
                v[i] -= rh.v[i];

        return *this;
    }

    size_t size()
    {
        return _size;
    }

    void print()
    {
        for (auto &el : v)
            cout << el << " ";

        cout << endl
             << endl;
    }
};

template <class T>
class myVec<myFun<T>>
{
private:
    size_t _size;
    vector<myFun<T>> v;
    vector<myFun<T>> fx;

private:
    myVec() = delete;

public:
    myVec(size_t _size, bool derivate) : _size(_size)
    {
        v.reserve(_size);

        if (derivate == DERIVATE)
        {
            for (int i = 0; i < _size; i++)
            {
                v.emplace_back(myVec::relu);    //TODO: implementar bien
                fx.emplace_back(myVec::fxrelu); //TODO: implementar bien
            }
        }
        else
        {
            for (int i = 0; i < _size; i++)
                v.emplace_back(myVec::relu); //TODO: implementar bien
        }
    }

    myVec(vector<size_t> &funs, bool derivate) : _size(_size)
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
                    v.emplace_back(myVec::relu);
                    fx.emplace_back(myVec::fxrelu);
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
                    v.emplace_back(myVec::relu);
                    break;
                }
            }
        }
    }

    myVec(const myVec &rh) : _size(rh._size), v(rh.v), fx(rh.fx) {}

    myVec(myVec &&rh) : _size(rh._size), v(move(rh.v)), fx(move(rh.fx)) {}

    myVec &operator=(const myVec &rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = rh.v;
            fx = rh.fx;
        }

        return *this;
    }

    myVec &operator=(myVec &&rh)
    {
        if (this != &rh)
        {
            _size = rh._size;
            v = move(rh.v);
            fx = move(rh.fx);
        }

        return *this;
    }

    static T relu(T &in)
    {

        if (in >= 0)
            return in;
        else
            return in / 8;
    }

    static T fxrelu(T &in)
    {

        if (in >= 0)
            return 1;
        else
            return (T)1 / 8;
    }

    myVec<T> calculate(myVec<T> &rh)
    {
#ifdef ASSERT
        if (_size != rh.size())
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh.size() << endl;
            return myVec<T>(0, CERO);
        }
#endif

        myVec<T> tmp(_size, CERO);

        for (int i = 0; i < _size; i++)
            tmp[i] = v[i](rh[i]);

        return tmp;
    }

    myVec<T> derivate(myVec<T> &rh)
    {
#ifdef ASSERT
        if (_size != rh.size())
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh.size() << endl;
            return myVec<T>(0, CERO);
        }
#endif

        myVec<T> tmp(_size, CERO);

        for (int i = 0; i < _size; i++)
            tmp[i] = fx[i](rh[i]);

        return tmp;
    }

    size_t size()
    {
        return _size;
    }
};

template <class T>
class myMatrix
{
    template <class U>
    friend myMatrix<U> make_from(myVec<U> &lh, myVec<U> &rh);

private:
    size_t _rows;
    size_t _cols;
    vector<myVec<T>> m;

private:
    myMatrix() = delete;

    myMatrix(size_t _rows, size_t _cols) : _rows(_rows), _cols(_cols)
    {
        m.reserve(_rows);
    }

public:
    myMatrix(size_t _rows, size_t _cols, int mode) : _rows(_rows), _cols(_cols)
    {
        m.reserve(_rows);

        switch (mode)
        {
        case RANDOM:
            for (int i = 0; i < _rows; i++)
                m.emplace_back(_cols, RANDOM);

            break;

        case CERO:
        default:
            for (int i = 0; i < _rows; i++)
                m.emplace_back(_cols, CERO);

            break;
        }
    }

    myMatrix(vector<vector<T>> &vecs) : _rows(vecs.size()), _cols(vecs[0].size())
    {
        m.reserve(_rows);

        for (int i = 0; i < _rows; i++)
            m.emplace_back(vecs[i]);
    }

    myMatrix(initializer_list<initializer_list<T>> l) : _rows(l.size()), _cols(l.begin()->size())
    {
        m.reserve(l.size());

        for (auto &list : l)
            m.emplace_back(list);
    }

    myMatrix(const myMatrix &rh) : _rows(rh._rows), _cols(rh._cols), m(rh.m) {}

    myMatrix(myMatrix &&rh) : _rows(rh._rows), _cols(rh._cols), m(move(rh.m)) {}

    myMatrix &operator=(const myMatrix &rh)
    {
        if (this != &rh)
        {
            _rows = rh._rows;
            _cols = rh._cols;
            m = rh.m;
        }

        return *this;
    }

    myMatrix &operator=(myMatrix &&rh)
    {
        if (this != &rh)
        {
            _rows = rh._rows;
            _cols = rh._cols;
            m = move(rh.m);
        }

        return *this;
    }

    myVec<T> &operator[](size_t row)
    {
#ifdef ASSERT
        if (row < _rows)
            return m[row];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
#else
        return m[row];
#endif
    }

    myMatrix operator*(myMatrix &rh)
    {
#ifdef ASSERT
        if (_cols != rh._rows)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
            return myMatrix(0, 0);
        }
#endif

        myMatrix tmp(_rows, rh._cols);

        for (int i = 0; i < _rows; i++)
        {
            myVec<T> row(rh._cols, CERO);

            for (int j = 0; j < rh._cols; j++)
            {
                T sum = 0;

                for (int k = 0; k < rh._rows; k++)
                    sum += (*this)[i][k] * rh[k][j];

                row[i] = sum;
            }

            tmp.m.emplace_back(row);
        }

        return tmp;
    }

    myMatrix operator+(myMatrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
            return myMatrix(0, 0);
        }
#endif

        myMatrix tmp(_rows, _cols);

        for (int i = 0; i < _rows; i++)
            tmp.m.emplace_back((*this)[i] + rh[i]);

        return tmp;
    }

    myMatrix &operator+=(myMatrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
        else
#endif
            for (int i = 0; i < _rows; i++)
                (*this)[i] += rh[i];

        return *this;
    }

    myMatrix operator-(myMatrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
            return myMatrix(0, 0);
        }
#endif

        myMatrix tmp(_rows, _cols);

        for (int i = 0; i < _rows; i++)
            tmp.m.emplace_back((*this)[i] - rh[i]);

        return tmp;
    }

    myMatrix &operator-=(myMatrix &rh)
    {
#ifdef ASSERT
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
        else
#endif
            for (int i = 0; i < _rows; i++)
                (*this)[i] -= rh[i];

        return *this;
    }

    myMatrix operator^(myVec<T> &rh)
    {
#ifdef ASSERT
        if (rh.size() != _rows)
        {
            cout << "invalid dimensions lh is " << _rows << " rh is " << rh.size() << endl;
            return myMatrix(0, 0);
        }
#endif

        myMatrix tmp = *this;

        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                tmp[i][j] *= rh[i];

        return tmp;
    }

    myMatrix &operator^=(myVec<T> &rh)
    {
#ifdef ASSERT
        if (rh.size() != _rows)
            cout << "invalid dimensions lh is " << _rows << " rh is " << rh.size() << endl;
        else
#endif
            for (int i = 0; i < _rows; i++)
                for (int j = 0; j < _cols; j++)
                    (*this)[i][j] *= rh[i];

        return *this;
    }

    size_t rows()
    {
        return _rows;
    }

    size_t cols()
    {
        return _cols;
    }

    void print()
    {
        for (auto &row : m)
            row.print();

        cout << endl;
    }
};

template <class U>
myVec<U> operator*(myVec<U> &vec, myMatrix<U> &matrix)
{
#ifdef ASSERT
    if (vec.size() != matrix.rows())
    {
        cout << "invalid dimensions lh is " << vec.size() << " rh is " << matrix.rows() << endl;
        return myVec<U>(0, CERO);
    }
#endif

    myVec<U> tmp(matrix.cols(), CERO);

    for (int j = 0; j < matrix.cols(); j++)
    {
        U sum = 0;

        for (int k = 0; k < matrix.rows(); k++)
            sum += vec[k] * matrix[k][j];

        tmp[j] = sum;
    }

    return tmp;
}

template <class U>
myVec<U> operator*(myMatrix<U> &matrix, myVec<U> &vec)
{
#ifdef ASSERT
    if (matrix.cols() != vec.size())
    {
        cout << "invalid dimensions lh is " << matrix.cols() << " rh is " << vec.size() << endl;
        return myVec<U>(0, CERO);
    }
#endif

    myVec<U> tmp(matrix.rows(), CERO);

    for (int j = 0; j < matrix.rows(); j++)
    {
        U sum = 0;

        for (int k = 0; k < matrix.cols(); k++)
            sum += vec[k] * matrix[j][k];

        tmp[j] = sum;
    }

    return tmp;
}

template <class U>
myMatrix<U> make_from(myVec<U> &lh, myVec<U> &rh)
{
    myMatrix<U> tmp(lh.size(), rh.size());

    for (int i = 0; i < lh.size(); i++)
    {
        myVec<U> row(rh.size(), CERO);

        for (int j = 0; j < rh.size(); j++)
            row[j] = lh[i] * rh[j];

        tmp.m.emplace_back(row);
    }

    return tmp;
}

#endif
