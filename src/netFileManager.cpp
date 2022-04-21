#include <netFileManager.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace net
{
    using namespace std;

    bool file_manager::load_net_structure(const string &file)
    {
        ifstream file_handler(PATH + '/' + file + ".csv", ios::in);

        if (file_handler.is_open())
        {
            string val, line;
            auto skip_lines = [&](int n_lines)
            {
                for (int i = 0; i < n_lines; i++)
                    getline(file_handler, line);
            };

            getline(file_handler, line);

            {
                stringstream s(line);
                getline(s, val, SEPARATOR);
                data.n_ins = (size_t)stoi(val);
                data.n_p_l.clear();

                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ') //* asegurar que no hay un espacio al final
                        data.n_p_l.push_back((size_t)stoi(val));

                data.n_layers = data.n_p_l.size();
            }

            skip_lines(1);
            getline(file_handler, line);

            {
                stringstream s(line);
                data.activation_type.clear();
                data.activation_type.reserve(data.n_layers);

                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')
                        data.activation_type.emplace_back(val == "R"
                                                              ? net::RELU
                                                          : val == "R2"
                                                              ? net::RELU2
                                                          : val == "R2S"
                                                              ? net::RELU2_SOFT_MAX
                                                              : net::SIGMOID);
            }

            file_handler.close();
            return true;
        }

        cout << RED << "unable to open file" << RESET << "\n";
        return false;
    }

    bool file_manager::load_net_structure(const string &file, bool file_reload)
    {
        if (net_structure_file != file || file_reload) //* load new file
        {
            bool succeeded = load_net_structure(file);

            if (succeeded && !file_reload) //* si es un nuevo archivo
                net_structure_file = file;

            return succeeded;
        }

        return true;
    }

    bool file_manager::load_net(const string &file)
    {
        ifstream file_handler(PATH + '/' + file + ".csv", ios::in);

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

            skip_lines(4);

            for (size_t i = 0; i < data.n_layers; i++)
            {
                if (i == 0)
                    data.params.emplace_back(data.n_p_l[i] * data.n_ins);
                else
                    data.params.emplace_back(data.n_p_l[i] * data.n_p_l[i - 1]);

                data.bias.emplace_back(data.n_p_l[i], 0);
                size_t rows = data.bias[i].size();
                size_t cols = data.params[i].size() / rows;

                for (size_t j = 0; j < rows; j++)
                {
                    getline(file_handler, line);
                    stringstream s(line);
                    size_t k = 0;

                    while (getline(s, val, SEPARATOR))
                        if (val[0] != ' ')
                            data.params[i][j * cols + k++] = stof(val);
                }

                skip_lines(1);
                getline(file_handler, line);
                stringstream s(line);
                size_t j = 0;

                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')
                        data.bias[i][j++] = stof(val);

                skip_lines(1);
            }

            file_handler.close();
            return true;
        }

        cout << RED << "unable to open file" << RESET << "\n";
        return false;
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
        ifstream file_handler(PATH + '/' + file + ".csv", ios::in);

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
            sets.set_ins.reserve(n_sets);
            sets.set_outs.reserve(n_sets);
            sets.set_ins.emplace_back(0, 0); //* load first set to get n_ins and n_layers, so we can reuse them later
            sets.set_outs.emplace_back(0, 0);
            size_t n_ins = 0;
            size_t n_outs = 0;

            skip_lines(1);
            getline(file_handler, line);

            {
                stringstream s(line);

                while (getline(s, val, SEPARATOR))
                {
                    if (val[0] != ' ')
                    {
                        n_ins++;
                        sets.set_ins[0].emplace_back(stof(val));
                    }
                }
            }

            getline(file_handler, line);

            {
                stringstream s(line);

                while (getline(s, val, SEPARATOR))
                {
                    if (val[0] != ' ')
                    {
                        n_outs++;
                        sets.set_outs[0].emplace_back(stof(val));
                    }
                }
            }

            skip_lines(1);

            for (int i = 1; i < n_sets; i++) //* remaining sets
            {
                sets.set_ins.emplace_back(n_ins, 0);
                sets.set_outs.emplace_back(n_outs, 0);

                getline(file_handler, line);

                {
                    stringstream s(line);
                    size_t j = 0;

                    while (getline(s, val, SEPARATOR))
                        if (val[0] != ' ')
                            sets.set_ins[i][j++] = stof(val);
                }

                getline(file_handler, line);

                {
                    stringstream s(line);
                    size_t j = 0;

                    while (getline(s, val, SEPARATOR))
                        if (val[0] != ' ')
                            sets.set_outs[i][j++] = stof(val);
                }

                skip_lines(1);
            }

            file_handler.close();
            return true;
        }

        cout << RED << "unable to open file" << RESET << "\n";
        return false;
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
        ofstream file_handler(PATH + '/' + file + ".csv", ios::out | ios::trunc);

        if (file_handler.is_open())
        {
            file_handler << n_data.n_ins << SEPARATOR;

            for (auto &i : n_data.n_p_l)
                file_handler << i << SEPARATOR;

            file_handler << "\n\n";

            for (auto &i : n_data.activation_type)
                file_handler << (i == net::RELU
                                     ? "R"
                                 : i == net::RELU2
                                     ? "R2"
                                 : i == net::RELU2_SOFT_MAX
                                     ? "R2S"
                                     : "S")
                             << SEPARATOR;

            file_handler << "\n\n";

            for (size_t i = 0; i < n_data.n_layers; i++)
            {
                size_t rows = n_data.bias[i].size();
                size_t cols = n_data.params[i].size() / rows;

                for (size_t j = 0; j < rows; j++)
                {
                    for (size_t k = 0; k < cols; k++)
                        file_handler << n_data.params[i][j * cols + k] << SEPARATOR;

                    file_handler << "\n";
                }

                file_handler << "\n";

                for (size_t j = 0; j < rows; j++)
                    file_handler << n_data.bias[i][j] << SEPARATOR;

                file_handler << "\n\n";
            }

            file_handler.close();
            return true;
        }

        cout << RED << "unable to open file" << RESET << "\n";
        return false;
    }

    bool file_manager::write_sets_to_file(const std::string &file, const net_sets &n_sets)
    {
        ofstream file_handler(PATH + '/' + file + ".csv", ios::out | ios::trunc);

        if (file_handler.is_open())
        {
            size_t data_size = n_sets.set_ins.size();
            file_handler << data_size << SEPARATOR;
            file_handler << "\n\n";

            for (size_t i = 0; i < data_size; i++)
            {
                for (auto &j : n_sets.set_ins[i])
                    file_handler << j << SEPARATOR;

                file_handler << "\n";

                for (auto &j : n_sets.set_outs[i])
                    file_handler << j << SEPARATOR;

                file_handler << "\n\n";
            }

            file_handler.close();
            return true;
        }

        cout << RED << "unable to open file" << RESET << "\n";
        return false;
    }

}