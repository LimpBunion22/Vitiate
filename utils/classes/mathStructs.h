#ifndef MATHSTRUCTS_H
#define MATHSTRUCTS_H

#include <vector>
#include <iostream>

using namespace std;

template <class T>
class myMatrix;

template <class T>
class myVec
{
    friend class myMatrix<T>;

    template <class U>
    friend myVec<U> operator*(myVec<U> vec, myMatrix<U> &matrix);

    template <class U>
    friend myVec<U> operator*(myMatrix<U> &matrix, myVec<U> vec);

    template <class U>
    friend myMatrix<U> makeFrom(myVec<U> &lh, myVec<U> &rh);

private:
    vector<T> v;
    size_t size;

private:
    myVec() {}
    myVec(const vector<T> &v) : v(v), size(v.size()) {}

public:
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
            v = move(rh.v);
            size = rh.size;
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
            cout << "invalid dimensions" << endl;
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
            cout << "invalid dimensions" << endl;
            return {};
        }

        vector<T> tmp;
        tmp.reserve(size);

        for (int i = 0; i < size; i++)
            tmp.emplace_back(v[i] * rh.v[i]);

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
class myMatrix
{
private:
    vector<vector<T>> m;
    size_t rows;
    size_t cols;

private:
    myMatrix() {}

public:
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
            this->m.emplace_back(vec->v);
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
            m = move(rh.m);
            rows = rh.rows;
            cols = rh.cols;
        }

        return *this;
    }

    T &operator()(size_t row, size_t col)
    {
        if (row < rows && col < cols)
            return m[row][col];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
    }

    myMatrix operator*(myMatrix &rh)
    {
        if (cols != rh.rows)
        {
            cout << "invalid dimensions" << endl;
            return {};
        }

        myMatrix tmp;
        tmp.m.reserve(rows);
        tmp.rows = rows;
        tmp.cols = rh.cols;

        vector<T> row;
        row.reserve(rh.cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rh.cols; j++)
            {
                T sum = 0;

                for (int k = 0; k < rh.rows; k++)
                    sum += (*this)(i, k) * rh(k, j);

                row.emplace_back(sum);
            }

            tmp.m.emplace_back(row);
            row.clear();
        }

        return tmp;
    }

    template <class U>
    friend myVec<U> operator*(myVec<U> vec, myMatrix<U> &matrix)
    {
        if (vec.size != matrix.rows)
        {
            cout << "invalid dimensions" << endl;
            return {};
        }

        vector<U> tmp;
        tmp.reserve(matrix.cols);

        for (int j = 0; j < matrix.cols; j++)
        {
            T sum = 0;

            for (int k = 0; k < matrix.rows; k++)
                sum += vec[k] * matrix(k, j);

            tmp.emplace_back(sum);
        }

        return tmp;
    }

    template <class U>
    friend myVec<U> operator*(myMatrix<U> &matrix, myVec<U> vec)
    {
        if (matrix.cols != vec.size)
        {
            cout << "invalid dimensions" << endl;
            return {};
        }

        vector<U> tmp;
        tmp.reserve(matrix.rows);

        for (int j = 0; j < matrix.rows; j++)
        {
            T sum = 0;

            for (int k = 0; k < matrix.cols; k++)
                sum += vec[k] * matrix(j, k);

            tmp.emplace_back(sum);
        }

        return tmp;
    }

    template <class U>
    friend myMatrix<U> makeFrom(myVec<U> &lh, myVec<U> &rh)
    {
        myMatrix<U> tmp;
        tmp.m.reserve(lh.size);
        tmp.rows = lh.size;
        tmp.cols = rh.size;

        vector<T> row;
        row.reserve(rh.size);

        for (int i = 0; i < lh.size; i++)
        {
            for (int j = 0; j < rh.size; j++)
                row.emplace_back(lh[i] * rh[j]);

            tmp.m.emplace_back(row);
            row.clear();
        }

        return tmp;
    }

    myMatrix operator^(myVec<T> &vec)
    {
        if (vec.size != rows)
        {
            cout << "invalid dimensions" << endl;
            return {};
        }

        myMatrix tmp = *this;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                tmp(i, j) *= vec[i];

        return tmp;
    }

    void print()
    {
        for (auto &row : m)
        {
            for (auto &c : row)
                cout << c << " ";

            cout << endl;
        }

        cout << endl;
    }
};

#endif