
import json

my_json = {}

def load_json(json_name):
    global my_json
    with open(json_name+".json", "r") as read_file:
        my_json = json.load(read_file)
    return

def save_json(json_name):
    global my_json
    with open(json_name+".json", "w") as write_file:
        json.dump(my_json, write_file)
    return

def read_main_param(param):
    global my_json
    if param in my_json:
        param_val = my_json[param]
    else:
        param_val = "Empty"
    return param_val

def write_main_param(param,param_val):
    global my_json
    my_json[param] = param_val
    return

def read_vector_param(param,ind):
    global my_json
    if param in my_json:
        param_val = my_json[param][ind]
    else:
        param_val = "Empty"
    return param_val

def write_vector_param(param, ind, param_val):
    global my_json
    if not(param in my_json):
        my_json[param] = []
    while ind+1>len(my_json[param]):
        my_json[param].append(param_val)
    my_json[param][ind] = param_val
    return
    