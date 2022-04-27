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
        const std::string _PATH;
        std::string _net_file = " ";
        std::string _set_file = " ";

    public:
        layout _net;
        set _set;

    private:
        bool _load_net(const std::string &file);
        bool _load_set(const std::string &file);

    public:
        file_manager(const std::string &path) : _PATH(path) {}

        bool load_net(const std::string &file, bool file_reload);
        bool load_set(const std::string &file, bool file_reload);
        bool write_net_to_file(const std::string &file, const layout &layout); //^ last row element also followed by separator char!!
        bool write_set_to_file(const std::string &file, const set &set);
    };
}
#endif