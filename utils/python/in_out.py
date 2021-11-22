
import os
import pickle

out_dict = {}

def add_key(key):
    global out_dict
    out_dict[key] = []
    return

def add_value(key, value):
    global out_dict
    out_dict[key].append(value)
    return

def save_dict(filename):
    global out_dict
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    with open('./results/' + filename + '.pkl', 'wb') as f:
        pickle.dump(out_dict, f)
    return