
import pandas as pd


df = None
aux_matrix = None

def load_csv(csv_name):
    global df
    df = pd.read_csv(csv_name+".csv",header=None)
    return

def save_csv(csv_name):
    global df
    global aux_matrix
    df = pd.DataFrame(aux_matrix)
    df.to_csv(csv_name+".csv")
    return

def read_value(x,y):
    global df
    return df[y][x]

def write_value(x,y,value):
    global df
    global aux_matrix
    if not(isinstance(aux_matrix,list)):
        aux_matrix = []
    while y>len(aux_matrix):
        aux_matrix.append(list())
    if x>len(aux_matrix[y]):
        while x>len(aux_matrix[y]):
            aux_matrix[y].append(list())
    aux_matrix[y][x] = value
    return

