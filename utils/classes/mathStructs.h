#ifndef MATHSTRUCTS_H
#define MATHSTRUCTS_H

#include <vector>
#include <iostream>

constexpr long MAX_RANGE = 10;
constexpr long MIN_RANGE = -10;
constexpr long RANGE = 2 * MAX_RANGE;
constexpr bool DERIVATE = true;
constexpr bool NOT_DERIVATE = false;
constexpr int RELU = 0;
constexpr int RELU2 = 1;
constexpr int SIGMOID = 2;
#define RANDOM nullptr

using namespace std;

template <class U>
using myFun = U (*)(U &in);

template <class T>
class myMatrix;

template <class T>
class myVec
{
    friend class myVec<myFun<T>>;
    friend class myMatrix<T>;

    template <class U>
    friend myVec<U> operator*(myVec<U> &vec, myMatrix<U> &matrix);

    template <class U>
    friend myVec<U> operator*(myMatrix<U> &matrix, myVec<U> &vec);

    template <class U>
    friend myMatrix<U> makeFrom(myVec<U> &lh, myVec<U> &rh);

private:
    size_t _size;
    vector<T> v;

private:
    myVec() = delete;

    myVec(size_t _size) : _size(_size)
    {
        v.reserve(_size);
    }

public:
    myVec(size_t _size, vector<T> *vals) : _size(_size)
    {
        if (vals)
            v = move(*vals);
        else
        {
            v.reserve(_size);

            for (int i = 0; i < _size; i++)
                v.emplace_back(T((float)random() / RAND_MAX * RANGE + MIN_RANGE));
        }
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
        if (i < _size)
            return v[i];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
    }

    T operator*(const myVec &rh)
    {
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return 0;
        }

        T sum = 0;

        for (int i = 0; i < _size; i++)
            sum += v[i] * rh.v[i];

        return sum;
    }

    myVec operator^(const myVec &rh)
    {
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec(0);
        }

        myVec<T> tmp(_size);

        for (int i = 0; i < _size; i++)
            tmp.v.emplace_back(v[i] * rh.v[i]);

        return tmp;
    }

    myVec &operator^=(const myVec &rh)
    {
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
        else
            for (int i = 0; i < _size; i++)
                v[i] *= rh.v[i];

        return *this;
    }

    myVec operator+(const myVec &rh)
    {
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec(0);
        }

        myVec<T> tmp(_size);

        for (int i = 0; i < _size; i++)
            tmp.v.emplace_back(v[i] + rh.v[i]);

        return tmp;
    }

    myVec &operator+=(const myVec &rh)
    {
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
        else
            for (int i = 0; i < _size; i++)
                v[i] += rh.v[i];

        return *this;
    }

    myVec operator-(const myVec &rh)
    {
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec(0);
        }

        myVec<T> tmp(_size);

        for (int i = 0; i < _size; i++)
            tmp.v.emplace_back(v[i] - rh.v[i]);

        return tmp;
    }

    myVec &operator-=(const myVec &rh)
    {
        if (_size != rh._size)
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
        else
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
    myVec(size_t _size, bool derivate, vector<size_t> *funs) : _size(_size)
    {
        v.reserve(_size);

        if (derivate == DERIVATE)
        {
            fx.reserve(_size);

            if (funs)
                for (int i = 0; i < _size; i++)
                {
                    switch ((*funs)[i])
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
            else
                for (int i = 0; i < _size; i++)
                {
                    v.emplace_back(myVec::relu);    //TODO: implementar bien
                    fx.emplace_back(myVec::fxrelu); //TODO: implementar bien
                }
        }
        else
        {
            if (funs)
                for (int i = 0; i < _size; i++)
                {
                    switch ((*funs)[i])
                    {
                    case RELU:
                    case RELU2:
                    case SIGMOID:
                    default:
                        v.emplace_back(myVec::relu);
                        break;
                    }
                }
            else
                for (int i = 0; i < _size; i++)
                    v.emplace_back(myVec::relu); //TODO: implementar bien
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
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec<T>(0);
        }

        myVec<T> tmp(_size);

        for (int i = 0; i < _size; i++)
            tmp.v.emplace_back(v[i](rh[i]));

        return tmp;
    }

    myVec<T> derivate(myVec<T> &rh)
    {
        if (_size != rh._size)
        {
            cout << "invalid dimensions lh is " << _size << " rh is " << rh._size << endl;
            return myVec<T>(0);
        }

        myVec<T> tmp(_size);

        for (int i = 0; i < _size; i++)
            tmp.v.emplace_back(fx[i](rh[i]));

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
    friend myVec<U> operator*(myVec<U> &vec, myMatrix<U> &matrix);

    template <class U>
    friend myVec<U> operator*(myMatrix<U> &matrix, myVec<U> &vec);

    template <class U>
    friend myMatrix<U> makeFrom(myVec<U> &lh, myVec<U> &rh);

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
    myMatrix(size_t _rows, size_t _cols, vector<vector<T>> *vecs) : _rows(_rows), _cols(_cols)
    {
        m.reserve(_rows);

        if (vecs)
        {
            for (int i = 0; i < _rows; i++)
                m.emplace_back(_cols, &(*vecs)[i]);
        }
        else
            for (int i = 0; i < _rows; i++)
                m.emplace_back(_cols, RANDOM);
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
            m = rh.m;
            _rows = rh._rows;
            _cols = rh._cols;
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
        if (row < _rows)
            return m[row];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
    }

    myMatrix operator*(myMatrix &rh)
    {
        if (_cols != rh._rows)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
            return myMatrix(0, 0);
        }

        myMatrix tmp(_rows, rh._cols);

        for (int i = 0; i < _rows; i++)
        {
            myVec<T> row(rh._cols);

            for (int j = 0; j < rh._cols; j++)
            {
                T sum = 0;

                for (int k = 0; k < rh._rows; k++)
                    sum += (*this)[i][k] * rh[k][j];

                row.v.emplace_back(sum);
            }

            tmp.m.emplace_back(row);
        }

        return tmp;
    }

    myMatrix operator+(myMatrix &rh)
    {
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
            return myMatrix(0, 0);
        }

        myMatrix tmp(_rows, _cols);

        for (int i = 0; i < _rows; i++)
            tmp.m.emplace_back((*this)[i] + rh[i]);

        return tmp;
    }

    myMatrix &operator+=(myMatrix &rh)
    {
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
        else
            for (int i = 0; i < _rows; i++)
                (*this)[i] += rh[i];

        return *this;
    }

    myMatrix operator-(myMatrix &rh)
    {
        if (_rows != rh._rows || _cols != rh._cols)
        {
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
            return myMatrix(0, 0);
        }

        myMatrix tmp(_rows, _cols);

        for (int i = 0; i < _rows; i++)
            tmp.m.emplace_back((*this)[i] - rh[i]);

        return tmp;
    }

    myMatrix &operator-=(myMatrix &rh)
    {
        if (_rows != rh._rows || _cols != rh._cols)
            cout << "invalid dimensions lh is " << _cols << " rh is " << rh._cols << endl;
        else
            for (int i = 0; i < _rows; i++)
                (*this)[i] -= rh[i];

        return *this;
    }

    myMatrix operator^(myVec<T> &rh)
    {
        if (rh._size != _rows)
        {
            cout << "invalid dimensions lh is " << _rows << " rh is " << rh._size << endl;
            return myMatrix(0, 0);
        }

        myMatrix tmp = *this;

        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                tmp[i][j] *= rh[i];

        return tmp;
    }

    myMatrix &operator^=(myVec<T> &rh)
    {
        if (rh._size != _rows)
            cout << "invalid dimensions lh is " << _rows << " rh is " << rh._size << endl;
        else
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
    if (vec._size != matrix._rows)
    {
        cout << "invalid dimensions lh is " << vec._size << " rh is " << matrix._rows << endl;
        return myVec<U>(0);
    }

    myVec<U> tmp(matrix._cols);

    for (int j = 0; j < matrix._cols; j++)
    {
        U sum = 0;

        for (int k = 0; k < matrix._rows; k++)
            sum += vec[k] * matrix[k][j];

        tmp.v.emplace_back(sum);
    }

    return tmp;
}

template <class U>
myVec<U> operator*(myMatrix<U> &matrix, myVec<U> &vec)
{
    if (matrix._cols != vec._size)
    {
        cout << "invalid dimensions lh is " << matrix._cols << " rh is " << vec._size << endl;
        return myVec<U>(0);
    }

    myVec<U> tmp(matrix._rows);

    for (int j = 0; j < matrix._rows; j++)
    {
        U sum = 0;

        for (int k = 0; k < matrix._cols; k++)
            sum += vec[k] * matrix[j][k];

        tmp.v.emplace_back(sum);
    }

    return tmp;
}

template <class U>
myMatrix<U> makeFrom(myVec<U> &lh, myVec<U> &rh)
{
    myMatrix<U> tmp(lh._size, rh._size);

    for (int i = 0; i < lh._size; i++)
    {
        myVec<U> row(rh._size);

        for (int j = 0; j < rh._size; j++)
            row.v.emplace_back(lh[i] * rh[j]);

        tmp.m.emplace_back(row);
    }

    return tmp;
}

#endif
