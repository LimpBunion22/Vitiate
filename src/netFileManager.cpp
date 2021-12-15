#include <netFileManager.h>
#include <netAbstract.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

void file_manager::load_net_structure(string name) // TODO reimplement
{
    ifstream file_handler(HOME + "/" + name + ".csv", ios::in);

    if (file_handler.is_open())
    {
        string val, line;

        getline(file_handler, line);
        stringstream s(line);

        getline(s, val, SEPARATOR);
        data.n_ins = (size_t)stoi(val);

        data.n_p_l.clear();

        while (getline(s, val, SEPARATOR))
            data.n_p_l.push_back((size_t)stoi(val));

        data.n_layers = data.n_p_l.size();

        file_handler.close();
    }
    else
        cout << "unable to open file\n";
}

void file_manager::load_net(string name) // TODO reimplement
{
    load_net_structure(name);
    ifstream file_handler(HOME + "/" + name + ".csv", ios::in);

    if (file_handler.is_open())
    {
        string val, line;
        data.params.clear();
        data.bias.clear();
        data.params.reserve(data.n_layers);
        data.bias.reserve(data.n_layers);
        auto skip_lines = [&](int n_lines)
        {
            for (int i = 0; i < n_lines; i++)
                getline(file_handler, line);
        };

        skip_lines(2);

        for (int i = 0; i < data.n_layers; i++)
        {
            if (i == 0)
                data.params.emplace_back(data.n_p_l[i], vector<DATA_TYPE>(data.n_ins, 0));
            else
                data.params.emplace_back(data.n_p_l[i], vector<DATA_TYPE>(data.n_p_l[i - 1], 0));

            data.bias.emplace_back(data.n_p_l[i], 0);

            for (int j = 0; j < data.params[i].size(); j++)
            {
                getline(file_handler, line);
                stringstream s(line);
                size_t k = 0;

                while (getline(s, val, SEPARATOR))
                    data.params[i][j][k++] = (DATA_TYPE)stod(val);
            }

            skip_lines(1);
            getline(file_handler, line);
            stringstream s(line);
            size_t j = 0;

            while (getline(s, val, SEPARATOR))
                data.bias[i][j++] = (DATA_TYPE)stod(val);

            skip_lines(1);
        }

        file_handler.close();
    }
    else
        cout << "unable to open file\n";
}

void file_manager::load_sets(string name)
{
    ifstream file_handler(HOME + "/" + name + ".csv", ios::in);

    if (file_handler.is_open())
    {
        string val, line;
        sets.set_ins.clear();
        sets.set_outs.clear();
        auto skip_lines = [&](int n_lines)
        {
            for (int i = 0; i < n_lines; i++)
                getline(file_handler, line);
        };

        getline(file_handler, line);
        stringstream s(line);
        getline(s, val, SEPARATOR);
        size_t n_sets = (size_t)stoi(val);
        skip_lines(1);
        sets.set_ins.reserve(n_sets);
        sets.set_outs.reserve(n_sets);

        for (int i = 0; i < n_sets; i++)
        {
            sets.set_ins.emplace_back(data.n_ins, 0);
            sets.set_outs.emplace_back(data.n_p_l[data.n_layers - 1], 0);

            getline(file_handler, line);

            {
                stringstream s(line);
                size_t j = 0;

                while (getline(s, val, SEPARATOR))
                    sets.set_ins[i][j++] = (DATA_TYPE)stod(val);
            }

            getline(file_handler, line);

            {
                stringstream s(line);
                size_t j = 0;

                while (getline(s, val, SEPARATOR))
                    sets.set_outs[i][j++] = (DATA_TYPE)stod(val);
            }

            skip_lines(1);
        }

        file_handler.close();
    }
    else
        cout << "unable to open file\n";
}

void file_manager::write_net_to_file(string name, net_abstract &net)
{
    ofstream file_handler(HOME + "/" + name + ".csv", ios::out | ios::trunc);

    if (file_handler.is_open())
    {
        net_data n_data = net.get_net_data();

        file_handler << n_data.n_ins << SEPARATOR;

        for (auto &i : n_data.n_p_l)
            file_handler << i << SEPARATOR;

        file_handler << "\n\n";

        for (int i = 0; i < n_data.n_layers; i++)
        {
            for (int j = 0; j < n_data.params[i].size(); j++)
            {
                for (int k = 0; k < n_data.params[i][j].size(); k++)
                    file_handler << n_data.params[i][j][k] << SEPARATOR;

                file_handler << "\n";
            }

            file_handler << "\n";

            for (int j = 0; j < n_data.bias[i].size(); j++)
                file_handler << n_data.bias[i][j] << SEPARATOR;

            file_handler << "\n\n";
        }

        file_handler.close();
    }
    else
        cout << "unable to open file\n";
}