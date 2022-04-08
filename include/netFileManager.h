#ifndef NETFILEMANAGER_H
#define NETFILEMANAGER_H

#include <defines.h>
#include <string>
#include <iostream>

namespace net
{
    constexpr char SEPARATOR = ',';
    constexpr bool RELOAD_FILE = true;
    constexpr bool REUSE_FILE = false;

    class file_manager
    {
    private:
        const std::string PATH;
        std::string net_file = " ";
        std::string net_structure_file = " ";
        std::string set_file = " ";

    public:
        net_data data;
        net_set set;

    private:
        bool load_net_structure(const std::string &file);
        bool load_net(const std::string &file);
        bool load_set(const std::string &file);

    public:
        file_manager(const std::string &path) : PATH(path) {}

        bool load_net_structure(const std::string &file, bool file_reload);
        bool load_net(const std::string &file, bool file_reload);
        bool load_set(const std::string &file, bool file_reload);
        bool write_net_to_file(const std::string &file, const net_data &n_data); //^ last row element also followed by separator char!!
        bool write_set_to_file(const std::string &file, const net_set &n_set);
    };
}
#endif