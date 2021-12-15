#ifndef NETFILEMANAGER_H
#define NETFILEMANAGER_H

#include <defines.h>
#include <string>

class net_abstract;

constexpr char SEPARATOR = ' ';

class file_manager
{
private:
    const std::string HOME = getenv("HOME") ? getenv("HOME") : ".";

public:
    net_data data;
    net_sets sets;

public:
    void load_net_structure(std::string name); // TODO REIMPLEMENT ALL
    void load_net(std::string name);
    void load_sets(std::string name);
    void write_net_to_file(std::string name, net_abstract &net); //^ last row element also followed by separator char!!
};

#endif