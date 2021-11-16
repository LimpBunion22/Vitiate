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
    vector<T> v;

public:
    size_t size;

private:
    myVec() = delete;

    myVec(size_t size) : size(size)
    {
        v.reserve(size);
    }

public:
    myVec(size_t size, vector<T> *vals) : size(size)
    {
        if (vals)
            v = move(*vals);
        else
        {
            v.reserve(size);

            for (int i = 0; i < size; i++)
                v.emplace_back(T(((float)random() / RAND_MAX * RANGE + MIN_RANGE)));
        }
    }

    myVec(initializer_list<T> l) : v(l), size(l.size()) {}

    myVec(const myVec &rh)
    {
        *this = rh;
    }

    myVec(myVec &&rh)
    {
        *this = move(rh);
    }

    myVec &operator=(const myVec &rh)
    {
        if (this != &rh)
        {
            v = rh.v;
            size = rh.size;
        }

        return *this;
    }

    myVec &operator=(myVec &&rh)
    {
        if (this != &rh)
        {
            size = rh.size;
            v = move(rh.v);
        }

        return *this;
    }

    T &operator[](int i)
    {
        if (i < size)
            return v[i];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
    }

    T operator*(const myVec &rh)
    {
        if (size != rh.size)
        {
            cout << "invalid dimensions lh is " << size << " rh is " << rh.size << endl;
            return 0;
        }

        T sum = 0;

        for (int i = 0; i < size; i++)
            sum += v[i] * rh.v[i];

        return sum;
    }

    myVec operator^(const myVec &rh)
    {
        if (size != rh.size)
        {
            cout << "invalid dimensions lh is " << size << " rh is " << rh.size << endl;
            return 0;
        }

        myVec<T> tmp(size);

        for (int i = 0; i < size; i++)
            tmp.v.emplace_back(v[i] * rh.v[i]);

        return tmp;
    }

    myVec operator+(const myVec &rh)
    {
        if (size != rh.size)
        {
            cout << "invalid dimensions lh is " << size << " rh is " << rh.size << endl;
            return 0;
        }

        myVec<T> tmp(size);

        for (int i = 0; i < size; i++)
            tmp.v.emplace_back(v[i] + rh.v[i]);

        return tmp;
    }

    myVec operator-(const myVec &rh)
    {
        if (size != rh.size)
        {
            cout << "invalid dimensions lh is " << size << " rh is " << rh.size << endl;
            return 0;
        }

        myVec<T> tmp(size);

        for (int i = 0; i < size; i++)
            tmp.v.emplace_back(v[i] - rh.v[i]);

        return tmp;
    }

    void print()
    {
        for (auto &el : v)
            cout << el << " ";

        cout << endl;
    }
};

template <class T>
class myVec<myFun<T>>
{
private:
    vector<myFun<T>> v;
    vector<myFun<T>> fx;

public:
    size_t size;

private:
    myVec() = delete;

public:
    myVec(size_t size, bool derivate, vector<size_t> *funs) : size(size)
    {
        v.reserve(size);

        if (derivate == DERIVATE)
        {
            fx.reserve(size);

            if (funs)
                for (int i = 0; i < size; i++)
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
                for (int i = 0; i < size; i++)
                {
                    v.emplace_back(myVec::relu);    //TODO: implementar bien
                    fx.emplace_back(myVec::fxrelu); //TODO: implementar bien
                }
        }
        else
        {
            if (funs)
                for (int i = 0; i < size; i++)
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
                for (int i = 0; i < size; i++)
                    v.emplace_back(myVec::relu); //TODO: implementar bien
        }
    }

    myVec(const myVec &rh)
    {
        *this = rh;
    }

    myVec(myVec &&rh)
    {
        *this = move(rh);
    }

    myVec &operator=(const myVec &rh)
    {
        if (this != &rh)
        {
            v = rh.v;
            fx = rh.fx;
            size = rh.size;
        }

        return *this;
    }

    myVec &operator=(myVec &&rh)
    {
        if (this != &rh)
        {
            size = rh.size;
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
        if (size != rh.size)
        {
            cout << "invalid dimensions lh is " << size << " rh is " << rh.size << endl;
            return myVec<T>(0, nullptr);
        }

        myVec<T> tmp(size);

        for (int i = 0; i < size; i++)
            tmp.v.emplace_back(v[i](rh[i]));

        return tmp;
    }

    myVec<T> derivate(myVec<T> &rh)
    {
        if (size != rh.size)
        {
            cout << "invalid dimensions lh is " << size << " rh is " << rh.size << endl;
            return myVec<T>(0, nullptr);
        }

        myVec<T> tmp(size);

        for (int i = 0; i < size; i++)
            tmp.v.emplace_back(fx[i](rh[i]));

        return tmp;
    }

    void print()
    {
        cout << "hola" << endl;
    }
};

template <class T>
class myMatrix
{
private:
    vector<myVec<T>> m;

public:
    size_t rows;
    size_t cols;

private:
    myMatrix() = delete;

    myMatrix(size_t rows, size_t cols) : rows(rows), cols(cols)
    {
        m.reserve(rows);
    }

public:
    myMatrix(size_t rows, size_t cols, vector<vector<T>> *vecs) : rows(rows), cols(cols)
    {
        m.reserve(rows);

        if (vecs)
        {
            for (int i = 0; i < rows; i++)
                m.emplace_back(cols, &(*vecs)[i]);
        }
        else
            for (int i = 0; i < rows; i++)
                m.emplace_back(cols, RANDOM);
    }

    myMatrix(initializer_list<initializer_list<T>> m)
    {
        rows = m.size();
        cols = m.begin()->size();
        this->m.reserve(m.size());

        for (auto &list : m)
        {
            this->m.emplace_back(list);
        }
    }

    myMatrix(vector<myVec<T> *> m)
    {
        rows = m.size();
        cols = m.back()->size;
        this->m.reserve(m.size());

        for (auto &vec : m)
        {
            this->m.emplace_back(*vec);
        }
    }

    myMatrix(const myMatrix &rh)
    {
        *this = rh;
    }

    myMatrix(myMatrix &&rh)
    {
        *this = move(rh);
    }

    myMatrix &operator=(const myMatrix &rh)
    {
        if (this != &rh)
        {
            m = rh.m;
            rows = rh.rows;
            cols = rh.cols;
        }

        return *this;
    }

    myMatrix &operator=(myMatrix &&rh)
    {
        if (this != &rh)
        {
            rows = rh.rows;
            cols = rh.cols;
            m = move(rh.m);
        }

        return *this;
    }

    myVec<T> &operator[](size_t row)
    {
        if (row < rows)
            return m[row];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
    }

    myMatrix operator*(myMatrix &rh)
    {
        if (cols != rh.rows)
        {
            cout << "invalid dimensions lh is " << cols << " rh is " << rh.cols << endl;
            return myMatrix(0, 0);
        }

        myMatrix tmp(rows, rh.cols);

        for (int i = 0; i < rows; i++)
        {
            myVec<T> row(rh.cols);

            for (int j = 0; j < rh.cols; j++)
            {
                T sum = 0;

                for (int k = 0; k < rh.rows; k++)
                    sum += (*this)[i][k] * rh[k][j];

                row.v.emplace_back(sum);
            }

            tmp.m.emplace_back(row);
        }

        return tmp;
    }

    template <class U>
    friend myVec<U> operator*(myVec<U> &vec, myMatrix<U> &matrix)
    {
        if (vec.size != matrix.rows)
        {
            cout << "invalid dimensions lh is " << vec.size << " rh is " << matrix.rows << endl;
            return 0;
        }

        myVec<U> tmp(matrix.cols);

        for (int j = 0; j < matrix.cols; j++)
        {
            T sum = 0;

            for (int k = 0; k < matrix.rows; k++)
                sum += vec[k] * matrix[k][j];

            tmp.v.emplace_back(sum);
        }

        return tmp;
    }

    template <class U>
    friend myVec<U> operator*(myMatrix<U> &matrix, myVec<U> &vec)
    {
        if (matrix.cols != vec.size)
        {
            cout << "invalid dimensions lh is " << matrix.cols << " rh is " << vec.size << endl;
            return 0;
        }

        myVec<U> tmp(matrix.rows);

        for (int j = 0; j < matrix.rows; j++)
        {
            T sum = 0;

            for (int k = 0; k < matrix.cols; k++)
                sum += vec[k] * matrix[j][k];

            tmp.v.emplace_back(sum);
        }

        return tmp;
    }

    template <class U>
    friend myMatrix<U> makeFrom(myVec<U> &lh, myVec<U> &rh)
    {
        myMatrix<U> tmp(lh.size, rh.size);

        for (int i = 0; i < lh.size; i++)
        {
            myVec<T> row(rh.size);

            for (int j = 0; j < rh.size; j++)
                row.v.emplace_back(lh[i] * rh[j]);

            tmp.m.emplace_back(row);
        }

        return tmp;
    }

    myMatrix operator^(myVec<T> &vec)
    {
        if (vec.size != rows)
        {
            cout << "invalid dimensions lh is " << rows << " rh is " << vec.size << endl;
            return myMatrix(0, 0);
        }

        myMatrix tmp = *this;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                tmp[i][j] *= vec[i];

        return tmp;
    }

    void print()
    {
        for (auto &row : m)
            row.print();

        cout << endl;
    }
};

#endif
