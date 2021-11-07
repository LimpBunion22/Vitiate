#include <stdio.h>
#include <Python.h>
#include <mathStructs.h>

//using namespace std;

int main()
{

    myVec<float> v1{3, 3};
    myVec<float> v2{2, 8};
    myVec<float> v3{1, 1};
    myVec<float> v4{2, 3, 1};
    myMatrix<float> m1{{1, 2, 3}, {2, 3, 4}, {6, 7, 8}};
    myMatrix<float> m2({&v1, &v2, &v3});

    cout << "Data -------------------------------" << endl;
    cout << "v1 is ";
    v1.print();
    cout << endl;
    cout << "v2 is ";
    v2.print();
    cout << endl;
    cout << "v3 is ";
    v3.print();
    cout << endl;
    cout << "v4 is ";
    v4.print();
    cout << endl;
    cout << "m1 is" << endl;
    m1.print();
    cout << "m2 is" << endl;
    m2.print();

    cout << "Operations--------------------------" << endl;
    cout << "v1*v2 equals " << v1 * v2 << endl;
    cout << endl;
    cout << "v1^v2 equals ";
    (v1 ^ v2).print();
    cout << endl;
    cout << "m1*m2 equals " << endl;
    (m1 * m2).print();
    cout << "v4*m2 equals ";
    (v4 * m2).print();
    cout << endl;
    cout << "m1*v4 equals ";
    (m1 * v4).print();
    cout << endl;
    cout << "v2 matrxix v4 equals " << endl;
    makeFrom(v2, v4).print();
    cout << "m1^v4 equals " << endl;
    (m1 ^ v4).print();

//-------------------------------------------------------
    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
    pName = PyUnicode_FromString((char*)"./utils/python/json_handler");
    pModule = PyImport_Import(pName);
    pFunc = PyObject_GetAttrString(pModule, (char*)"test_python");
    pArgs = PyTuple_Pack(1, PyUnicode_FromString((char*)"there!"));
    pValue = PyObject_CallObject(pFunc, pArgs);
    auto result = _PyUnicode_AsString(pValue);
    std::cout << result << std::endl;

	return 0;
}

