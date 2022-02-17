#include <netFileManager.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace net
{
    using namespace std;

    bool file_manager::load_net_structure(const string &file)
    {
        ifstream file_handler(PATH + "/" + file + ".csv", ios::in);

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
            // cout << "file read!\n";
            return true;
        }
        else
        {
            cout << "unable to open file\n";
            return false;
        }
    }

    bool file_manager::load_net_structure(const string &file, bool file_reload)
    {
        if (net_structure_file != file || file_reload) //* load new file
        {
            bool succeeded = load_net_structure(file);

            if (succeeded && !file_reload)
                net_structure_file = file;

            return succeeded;
        }

        return true;
    }

    bool file_manager::load_net(const string &file)
    {
        ifstream file_handler(PATH + "/" + file + ".csv", ios::in);

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
            // cout << "file read!\n";
            return true;
        }
        else
        {
            cout << "unable to open file\n";
            return false;
        }
    }

    bool file_manager::load_net(const string &file, bool file_reload)
    {
        if (net_file != file || file_reload) //* load new file
        {
            bool succeeded = load_net_structure(file, file_reload);

            if (succeeded)
                succeeded = load_net(file);
            else
                return false;

            if (succeeded && !file_reload) //* si es un nuevo archivo
                net_file = file;

            return succeeded;
        }

        return true;
    }

    bool file_manager::load_sets(const string &file)
    {
        ifstream file_handler(PATH + "/" + file + ".csv", ios::in);

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
            //* load first set to get n_ins and n_layers, so we can reuse them later
            sets.set_ins.emplace_back(0, 0);
            sets.set_outs.emplace_back(0, 0);
            size_t n_ins = 0;
            size_t n_outs = 0;

            getline(file_handler, line);

            {
                stringstream s(line);

                while (getline(s, val, SEPARATOR))
                {
                    n_ins++;
                    sets.set_ins[0].emplace_back((DATA_TYPE)stod(val));
                }
            }

            getline(file_handler, line);

            {
                stringstream s(line);

                while (getline(s, val, SEPARATOR))
                {
                    n_outs++;
                    sets.set_outs[0].emplace_back((DATA_TYPE)stod(val));
                }
            }

            skip_lines(1);
            //* remaining sets
            for (int i = 1; i < n_sets; i++)
            {
                sets.set_ins.emplace_back(n_ins, 0);
                sets.set_outs.emplace_back(n_outs, 0);

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
            // cout << "file read!\n";
            return true;
        }
        else
        {
            cout << "unable to open file\n";
            return false;
        }
    }

    bool file_manager::load_sets(const string &file, bool file_reload)
    {
        if (sets_file != file || file_reload) //* load new file
        {
            bool succeeded = load_sets(file);

            if (succeeded && !file_reload)
                sets_file = file;

            return succeeded;
        }

        return true;
    }

    bool file_manager::write_net_to_file(const string &file, const net_data &n_data)
    {
        ofstream file_handler(PATH + "/" + file + ".csv", ios::out | ios::trunc);

        if (file_handler.is_open())
        {
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
            // cout << "file written!\n";
            return true;
        }
        else
        {
            cout << "unable to open file\n";
            return false;
        }
    }
}