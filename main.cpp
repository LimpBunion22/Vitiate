#include <stdio.h>
#include <network.h>
#include <time.h>
#include <vector>

using namespace std;

#define ASSERT

int main()
{
    srand(time(NULL));

    myVec<float> ins = {2.0f, 1.0f};
    myVec<float> set_outs = {0, 0, 0, 0, 0};
    myVec<size_t> neurons_per_layer = {3, 4, 5};

    vector<vector<vector<float>>> params =
        {{{1, 2},
          {-0.1, 0.1},
          {0, 0.1}},
         {{1, 2, -0.1},
          {1, -0.1, -0.1},
          {1, 2, -0.1},
          {1, 0.1, 0.1}},
         {{1, 2, -0.1, 0.1},
          {1, 2, -0.1, 0.1},
          {0.1, 2, -1, 0.1},
          {-0.1, 2, -1, 0.1},
          {2, 2, 0.7, 0.1}}};

    vector<vector<float>> bias =
        {{-0.1, 0, -0.1},
         {0.1, 0.1, 0.1, 0.1},
         {0, 0, 0, 0, 0}};

    network<float> my_network(ins.size(), neurons_per_layer, DERIVATE, &params, &bias);
    network<float>::fx_container my_container(my_network);
    my_network.init_gradient();

    int iterations = 10;

    for (int i = 0; i < iterations; i++)
    {
        my_network.gradient(ins, set_outs, my_container);
        my_container.normalize_1();
        my_network.update_params(my_container);
    }

    my_network.print_inner_vals();

    return 0;
}
