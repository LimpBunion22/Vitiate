#include <pybind11/pybind11.h>
#include <stdio.h>
#include <mathStructs.h>
#include <pybind11/embed.h>



namespace py = pybind11;
using namespace std;

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
   
    py::scoped_interpreter guard{};

    py::exec(R"(
        kwargs = dict(name="World", number=42)
        message = "Hello, {name}! The answer is {number}".format(**kwargs)
        print(message)
    )");
    
	return 0;
}

