#include <netFileManager.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace net
{
    bool file_manager::_load_net(const std::string &file)
    {
        std::ifstream file_handler(_PATH + '/' + file + ".csv", std::ios::in);

        if (file_handler.is_open())
        {
            _net.activation.clear(); // clear prev stored vals
            _net.n_p_l.clear();
            _net.param_bias.clear();

            std::string val, line;
            auto skip_lines = [&](int n_lines)
            {
                for (int i = 0; i < n_lines; i++)
                    getline(file_handler, line);
            };

            skip_lines(1); // structure info
            getline(file_handler, line);

            {
                std::stringstream s(line);
                getline(s, val, SEPARATOR);
                _net.input_size = stoi(val); // input size

                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')                      //* get rid of any final blank space
                        _net.n_p_l.emplace_back(stoi(val)); // n_p_l
            }

            skip_lines(2); // activations info
            getline(file_handler, line);

            {
                std::stringstream s(line);

                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')
                        _net.activation.emplace_back(val == "R"
                                                         ? net::RELU
                                                     : val == "R2"
                                                         ? net::RELU2
                                                     : val == "R2S"
                                                         ? net::RELU2_SOFT_MAX
                                                         : net::SIGMOID); // activations
            }

            size_t net_size = 0; // get net_size

            for (size_t i = 0; i < _net.n_p_l.size(); i++)
            {
                if (i == 0)
                    net_size += _net.n_p_l[i] * _net.input_size + _net.n_p_l[i]; // params+bias
                else
                    net_size += _net.n_p_l[i] * _net.n_p_l[i - 1] + _net.n_p_l[i];
            }

            _net.param_bias.reserve(net_size);

            skip_lines(2); // net values info
            getline(file_handler, line);

            {
                std::stringstream s(line);

                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')
                        _net.param_bias.emplace_back(stof(val)); // params +bias
            }

            file_handler.close();
            return true;
        }

        std::cout << RED << "unable to open file" << RESET << "\n";
        return false;
    }

    bool file_manager::load_net(const std::string &file, bool file_reload)
    {
        if (_net_file != file || file_reload) //* if new data to be loaded
        {
            bool succeeded = _load_net(file);

            if (succeeded && !file_reload) //* if new file name
                _net_file = file;

            return succeeded;
        }

        return true;
    }

    bool file_manager::_load_set(const std::string &file)
    {
        std::ifstream file_handler(_PATH + '/' + file + ".csv", std::ios::in);

        if (file_handler.is_open())
        {
            _set.input_data.clear(); // clear prev stored vals
            _set.output_data.clear();
            _set.labels.clear();

            std::string val, line;
            auto skip_lines = [&](int n_lines)
            {
                for (int i = 0; i < n_lines; i++)
                    getline(file_handler, line);
            };

            size_t samples_num, out_size, in_size;

            skip_lines(1); // samples info
            getline(file_handler, line);

            {
                std::stringstream s(line);
                getline(s, val, SEPARATOR);
                _set.data_num = stoi(val); // number of samples
                samples_num = (size_t)_set.data_num;
            }

            skip_lines(2); // sizes info
            getline(file_handler, line);

            {
                std::stringstream s(line);
                getline(s, val, SEPARATOR);
                out_size = stoi(val);
                getline(s, val, SEPARATOR);
                in_size = stoi(val);
            }

            _set.input_data.reserve(samples_num * out_size); // reserve mem
            _set.output_data.reserve(samples_num * in_size);
            _set.labels.reserve(samples_num);

            skip_lines(2); // sample order info
            getline(file_handler, line);

            {
                std::stringstream s(line); // inputs
                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')
                        _set.input_data.emplace_back(stof(val));
            }

            skip_lines(1); // blank
            getline(file_handler, line);

            {
                std::stringstream s(line); // outputs
                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')
                        _set.output_data.emplace_back(stof(val));
            }

            skip_lines(1); // blank
            getline(file_handler, line);

            {
                std::stringstream s(line); // labels
                while (getline(s, val, SEPARATOR))
                    if (val[0] != ' ')
                        _set.labels.emplace_back(stoi(val));
            }

            file_handler.close();
            return true;
        }

        std::cout << RED << "unable to open file" << RESET << "\n";
        return false;
    }

    bool file_manager::load_set(const std::string &file, bool file_reload)
    {
        if (_set_file != file || file_reload) //* load new file
        {
            bool succeeded = _load_set(file);

            if (succeeded && !file_reload)
                _set_file = file;

            return succeeded;
        }

        return true;
    }

    bool file_manager::write_net_to_file(const std::string &file, const layout &layout)
    {
        std::ofstream file_handler(_PATH + '/' + file + ".csv", std::ios::out | std::ios::trunc);

        if (file_handler.is_open())
        {
            file_handler << "//input size, neurons per layer\n";
            file_handler << layout.input_size << SEPARATOR;

            for (auto &i : layout.n_p_l)
                file_handler << i << SEPARATOR;

            file_handler << "\n\n//activations\n";

            for (auto &i : layout.activation)
                file_handler << (i == net::RELU
                                     ? "R"
                                 : i == net::RELU2
                                     ? "R2"
                                 : i == net::RELU2_SOFT_MAX
                                     ? "R2S"
                                     : "S")
                             << SEPARATOR;

            file_handler << "\n\n//params + bias (interleaved)\n";

            for (auto &i : layout.param_bias)
                file_handler << i << SEPARATOR;

            file_handler.close();
            return true;
        }

        std::cout << RED << "unable to open file" << RESET << "\n";
        return false;
    }

    bool file_manager::write_set_to_file(const std::string &file, const set &set)
    {
        std::ofstream file_handler(_PATH + '/' + file + ".csv", std::ios::out | std::ios::trunc);

        if (file_handler.is_open())
        {
            file_handler << "//samples num\n";
            file_handler << set.data_num << SEPARATOR;

            file_handler << "\n\n//input & output size (label size = 1)\n";
            file_handler << ((int)set.input_data.size() / set.data_num) << SEPARATOR
                         << ((int)set.output_data.size() / set.data_num) << SEPARATOR;

            file_handler << "\n\n//inputs, outputs & labels\n";

            for (auto &i : set.input_data)
                file_handler << i << SEPARATOR;

            file_handler << "\n\n";

            for (auto &i : set.output_data)
                file_handler << i << SEPARATOR;

            file_handler << "\n\n";

            for (auto &i : set.labels)
                file_handler << i << SEPARATOR;

            file_handler.close();
            return true;
        }

        std::cout << RED << "unable to open file" << RESET << "\n";
        return false;
    }
}